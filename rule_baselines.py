from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Set

import numpy as np

try:
    from smartmap_mdv.baselines import normalize_text as shared_normalize_text
    from smartmap_mdv.baselines import tokenize_text as shared_tokenize_text
except Exception:
    shared_normalize_text = None
    shared_tokenize_text = None


TYPE_GROUPS = {
    "ip": {"ip", "ipaddr", "ipaddress", "ipv4", "ipv6"},
    "port": {"port"},
    "integer": {"integer", "int", "long", "short", "number", "numeric", "bigint", "smallint"},
    "float": {"float", "double", "decimal", "real"},
    "datetime": {"datetime", "timestamp", "time", "date"},
    "boolean": {"bool", "boolean"},
    "string": {"string", "str", "text", "keyword", "varchar", "char"},
    "array": {"array", "list"},
    "object": {"object", "dict", "json", "map"},
}

TYPE_COMPAT = {
    ("ip", "ip"): 1.0,
    ("port", "port"): 1.0,
    ("integer", "integer"): 1.0,
    ("float", "float"): 1.0,
    ("datetime", "datetime"): 1.0,
    ("boolean", "boolean"): 1.0,
    ("string", "string"): 1.0,
    ("array", "array"): 1.0,
    ("object", "object"): 1.0,
    ("integer", "port"): 0.8,
    ("port", "integer"): 0.8,
    ("integer", "float"): 0.65,
    ("float", "integer"): 0.65,
    ("string", "datetime"): 0.35,
    ("datetime", "string"): 0.35,
    ("string", "ip"): 0.30,
    ("ip", "string"): 0.30,
    ("string", "port"): 0.25,
    ("port", "string"): 0.25,
    ("string", "boolean"): 0.20,
    ("boolean", "string"): 0.20,
}

SRC_TOKENS = {"source", "src", "client", "origin", "sender", "attacker"}
DST_TOKENS = {"destination", "dst", "dest", "target", "server", "victim", "receiver"}

SLOT_PATTERNS = {
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PORT": re.compile(r"(?::|\b)(?:[1-9][0-9]{0,4})\b"),
    "TIME": re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\b"),
    "URL": re.compile(r"\bhttps?://[^\s]+", re.I),
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "MAC": re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I),
    "HASH": re.compile(r"\b[a-f0-9]{32,128}\b", re.I),
    "DOMAIN": re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b"),
}

SLOT_HINTS = {
    "ip": "IP",
    "address": "IP",
    "port": "PORT",
    "time": "TIME",
    "timestamp": "TIME",
    "date": "TIME",
    "url": "URL",
    "uri": "URL",
    "domain": "DOMAIN",
    "dns": "DOMAIN",
    "email": "EMAIL",
    "mac": "MAC",
    "hash": "HASH",
    "md5": "HASH",
    "sha1": "HASH",
    "sha256": "HASH",
    "user": "USER",
    "account": "USER",
    "host": "HOST",
    "hostname": "HOST",
    "event": "EVENT",
    "attack": "EVENT",
    "action": "ACTION",
    "policy": "POLICY",
    "rule": "POLICY",
    "status": "STATUS",
    "result": "STATUS",
}

COMPACT_SPLITS = {
    "srcip": ["source", "ip"],
    "sourceip": ["source", "ip"],
    "sip": ["source", "ip"],
    "dstip": ["destination", "ip"],
    "destip": ["destination", "ip"],
    "destinationip": ["destination", "ip"],
    "dip": ["destination", "ip"],
    "srcport": ["source", "port"],
    "sport": ["source", "port"],
    "dstport": ["destination", "port"],
    "dport": ["destination", "port"],
    "eventid": ["event", "id"],
    "ruleid": ["rule", "id"],
    "policyid": ["policy", "id"],
    "userid": ["user", "id"],
    "username": ["user", "name"],
    "hostname": ["host", "name"],
    "filepath": ["file", "path"],
    "filename": ["file", "name"],
    "filetype": ["file", "type"],
    "macaddr": ["mac", "address"],
    "macaddress": ["mac", "address"],
    "ipaddr": ["ip", "address"],
    "ipaddress": ["ip", "address"],
}


def _normalize_text(text: str) -> str:
    if shared_normalize_text is not None:
        return shared_normalize_text(text)
    text = str(text or "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"[\[\]\(\)\{\},:;/\\|]+", " ", text)
    text = re.sub(r"[\._$\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize_text(text: str) -> List[str]:
    if shared_tokenize_text is not None:
        return [tok for tok in shared_tokenize_text(text, strong=True) if tok]
    norm = _normalize_text(text)
    return [tok for tok in norm.split(" ") if tok]


def _normalize_token_set(text: str) -> Set[str]:
    toks = _tokenize_text(text)
    out: List[str] = []
    for tok in toks:
        out.append(tok)
        if tok in COMPACT_SPLITS:
            out.extend(COMPACT_SPLITS[tok])
        if tok.startswith("src") and tok != "src":
            out.extend(["source", tok[3:]])
        if tok.startswith("dst") and tok != "dst":
            out.extend(["destination", tok[3:]])
        if tok.startswith("dest") and tok != "dest":
            out.extend(["destination", tok[4:]])
        if tok.startswith("sport") and tok != "sport":
            out.extend(["source", "port"])
        if tok.startswith("dport") and tok != "dport":
            out.extend(["destination", "port"])
    return {tok for tok in out if tok}


def _compact_norm(text: str) -> str:
    return "".join(_tokenize_text(text))


def _simple_norm_name(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _simple_path_tokens(text: str) -> Set[str]:
    text = str(text or "").lower()
    text = re.sub(r"[_\./\-]+", " ", text)
    return {tok for tok in text.split() if tok}


def _type_base(x: Any) -> str:
    if isinstance(x, dict):
        x = x.get("base", "")
    t = str(x or "").lower().strip()
    t = re.sub(r"[^a-z0-9]+", "", t)
    for canon, members in TYPE_GROUPS.items():
        if t in members:
            return canon
    return t


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(min(len(a), len(b)), 1)


def _char_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _field_desc_text(rec: Dict[str, Any]) -> str:
    desc = rec.get("description") or rec.get("desc") or ""
    ex = rec.get("example", "")
    if not ex and rec.get("examples"):
        exs = rec.get("examples")
        if isinstance(exs, list):
            ex = " ".join(str(x) for x in exs if x)
        else:
            ex = str(exs or "")
    return f"{desc} {ex}".strip()


def _slot_set(text: str, tokens: Set[str]) -> Set[str]:
    text = str(text or "")
    slots = set()
    for slot, pat in SLOT_PATTERNS.items():
        if pat.search(text):
            slots.add(slot)
    for tok in tokens:
        slot = SLOT_HINTS.get(tok)
        if slot:
            slots.add(slot)
    return slots


def _path_list(text: str) -> List[str]:
    return [tok for tok in _tokenize_text(text) if tok]


def _prefix_suffix_bonus(src_toks: List[str], tgt_toks: List[str]) -> float:
    if not src_toks or not tgt_toks:
        return 0.0
    score = 0.0
    if src_toks[-1] == tgt_toks[-1]:
        score += 0.30
    if src_toks[0] == tgt_toks[0]:
        score += 0.15
    if len(src_toks) >= 2 and len(tgt_toks) >= 2 and src_toks[-2:] == tgt_toks[-2:]:
        score += 0.35
    if len(src_toks) >= 2 and len(tgt_toks) >= 2 and src_toks[:2] == tgt_toks[:2]:
        score += 0.20
    return score


def _cross_component_bonus(
    s_name_tokens: Set[str],
    t_name_tokens: Set[str],
    s_path_tokens: Set[str],
    t_path_tokens: Set[str],
) -> float:
    score = 0.0
    cross_j = max(_jaccard(s_name_tokens, t_path_tokens), _jaccard(t_name_tokens, s_path_tokens))
    cross_o = max(_overlap(s_name_tokens, t_path_tokens), _overlap(t_name_tokens, s_path_tokens))
    if cross_j >= 0.80:
        score += 0.55
    elif cross_j >= 0.50:
        score += 0.30
    elif cross_j > 0.0:
        score += 0.10
    if cross_o >= 0.80:
        score += 0.28
    elif cross_o >= 0.50:
        score += 0.15
    return score


def _polarity_bonus(a: Set[str], b: Set[str]) -> float:
    a_src = bool(a & SRC_TOKENS)
    a_dst = bool(a & DST_TOKENS)
    b_src = bool(b & SRC_TOKENS)
    b_dst = bool(b & DST_TOKENS)
    if (a_src and b_src) or (a_dst and b_dst):
        return 0.35
    return 0.0


def _polarity_penalty(a: Set[str], b: Set[str]) -> float:
    a_src = bool(a & SRC_TOKENS)
    a_dst = bool(a & DST_TOKENS)
    b_src = bool(b & SRC_TOKENS)
    b_dst = bool(b & DST_TOKENS)
    if (a_src and b_dst) or (a_dst and b_src):
        return 1.00
    return 0.0


def _type_compat(src_type: str, tgt_type: str) -> float:
    if not src_type or not tgt_type:
        return 0.0
    return TYPE_COMPAT.get((src_type, tgt_type), 0.0)


def _build_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    name = str(rec.get("name", "") or "")
    path = str(rec.get("path", "") or "")
    desc = _field_desc_text(rec)
    type_base = _type_base(rec.get("type", ""))

    name_tokens = _normalize_token_set(name)
    path_tokens = _normalize_token_set(path)
    desc_tokens = _normalize_token_set(desc)
    combined_tokens = name_tokens | path_tokens
    full_tokens = combined_tokens | desc_tokens

    slots = _slot_set(" ".join([name, path, desc]), full_tokens)

    return {
        "name_norm": _compact_norm(name),
        "path_norm": _compact_norm(path),
        "name_tokens": name_tokens,
        "path_tokens": path_tokens,
        "desc_tokens": desc_tokens,
        "combined_tokens": combined_tokens,
        "full_tokens": full_tokens,
        "path_list": _path_list(path),
        "name_list": _path_list(name),
        "slots": slots,
        "type": type_base,
    }


def _rule_score_heuristic_support(src: Dict[str, Any], tgt: Dict[str, Any]) -> float:
    score = 0.0
    src_name = _compact_norm(src.get("name", ""))
    tgt_name = _compact_norm(tgt.get("name", ""))
    src_type = _type_base(src.get("type", ""))
    tgt_type = _type_base(tgt.get("type", ""))
    src_path = _normalize_token_set(src.get("path", ""))
    tgt_path = _normalize_token_set(tgt.get("path", ""))
    path_j = _jaccard(src_path, tgt_path)

    if src_name and src_name == tgt_name:
        score += 0.060
    elif src_name and tgt_name and src_name[:4] == tgt_name[:4]:
        score += 0.008

    if src_type and tgt_type and src_type == tgt_type:
        score += 0.010

    # PATH remains a weak tie-breaker when some name signal exists.
    if score > 0.0:
        if path_j >= 0.80:
            score += 0.010
        elif path_j >= 0.50:
            score += 0.004

    return score


def rule_score_heuristic(src: Dict[str, Any], tgt: Dict[str, Any]) -> float:
    src_name = _compact_norm(src.get("name", ""))
    tgt_name = _compact_norm(tgt.get("name", ""))
    src_type = _type_base(src.get("type", ""))
    tgt_type = _type_base(tgt.get("type", ""))
    src_path = _normalize_token_set(src.get("path", ""))
    tgt_path = _normalize_token_set(tgt.get("path", ""))
    path_j = _jaccard(src_path, tgt_path)
    name_exact = bool(src_name and tgt_name and src_name == tgt_name)
    name_prefix = bool(
        src_name and tgt_name and len(src_name) >= 4 and len(tgt_name) >= 4 and src_name[:4] == tgt_name[:4]
    )
    type_exact = bool(src_type and tgt_type and src_type == tgt_type)

    # Keep heuristic intentionally coarse so it behaves like a weak rule
    # baseline rather than a lexical retriever.
    if name_exact and type_exact and path_j >= 0.50:
        return 0.28
    if name_exact and type_exact and path_j >= 0.20:
        return 0.08
    if name_exact and path_j >= 0.80:
        return 0.01
    if name_exact:
        return 0.0

    if name_prefix and type_exact and path_j >= 0.80:
        return 0.006

    # Keep a very weak fallback path so top-k grows more naturally than top-1
    # without turning the heuristic into a strong lexical baseline again.
    if type_exact and path_j >= 0.80:
        return 0.001
    if name_prefix:
        return 0.004
    if path_j >= 0.80:
        return 0.002

    return 0.0


def _score_from_features_raw(src_feat: Dict[str, Any], tgt_feat: Dict[str, Any]) -> float:
    name_exact = 1.0 if src_feat["name_norm"] and src_feat["name_norm"] == tgt_feat["name_norm"] else 0.0
    name_partial = 1.0 if (
        src_feat["name_norm"] and tgt_feat["name_norm"] and (
            src_feat["name_norm"] in tgt_feat["name_norm"] or tgt_feat["name_norm"] in src_feat["name_norm"]
        )
    ) else 0.0

    name_j = _jaccard(src_feat["name_tokens"], tgt_feat["name_tokens"])
    name_o = _overlap(src_feat["name_tokens"], tgt_feat["name_tokens"])
    path_j = _jaccard(src_feat["path_tokens"], tgt_feat["path_tokens"])
    path_o = _overlap(src_feat["path_tokens"], tgt_feat["path_tokens"])
    type_score = _type_compat(src_feat["type"], tgt_feat["type"])

    score = 0.0
    score += 0.095 * name_exact
    score += 0.012 * name_partial

    if name_j >= 0.80:
        score += 0.035
    elif name_j >= 0.50:
        score += 0.015

    if name_o >= 0.80:
        score += 0.012
    elif name_o >= 0.50:
        score += 0.004

    # Without strong NAME agreement, the rule baseline should stay weak.
    has_name_signal = bool(name_exact or name_j >= 0.80 or name_partial)
    if not has_name_signal:
        # Allow only very narrow rescue for strong type+path agreement.
        if not (type_score >= 1.0 and path_j >= 0.80):
            return 0.0

    if has_name_signal:
        if path_j >= 0.80:
            score += 0.008
        elif path_j >= 0.50:
            score += 0.003

        if path_o >= 0.80:
            score += 0.0015

    # Intentionally keep the enhanced rule baseline narrow. The richer slot,
    # cross-component, char-similarity, and prefix/suffix bonuses were making
    # it behave too much like a lexical retriever instead of a lightweight rule baseline.

    if type_score >= 1.0:
        score += 0.006
    elif type_score >= 0.80:
        score += 0.002
    elif type_score >= 0.30:
        score += 0.0008
    elif src_feat["type"] and tgt_feat["type"]:
        score -= 0.18

    score -= 1.15 * _polarity_penalty(src_feat["combined_tokens"], tgt_feat["combined_tokens"])
    return max(score, 0.0)


def rule_score_enhanced(src: Dict[str, Any], tgt: Dict[str, Any]) -> float:
    src_feat = _build_features(src)
    tgt_feat = _build_features(tgt)
    heuristic = _rule_score_heuristic_support(src, tgt)
    strong = _score_from_features_raw(src_feat, tgt_feat)
    # Blend the richer rule score with the classic heuristic so the enhanced
    # baseline stays meaningfully stronger than heuristic without overshooting
    # into lexical-baseline territory.
    return (0.78 * strong) + (0.10 * heuristic)


def pairwise_rule_scores_heuristic(
    A: Dict[str, Dict[str, Any]],
    B: Dict[str, Dict[str, Any]],
    idsA: List[str],
    idsB: List[str],
) -> np.ndarray:
    S = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
    for i, a in enumerate(idsA):
        src = A[a]
        for j, b in enumerate(idsB):
            tgt = B[b]
            S[i, j] = rule_score_heuristic(src, tgt)
    return S


def pairwise_rule_scores_enhanced(
    A: Dict[str, Dict[str, Any]],
    B: Dict[str, Dict[str, Any]],
    idsA: List[str],
    idsB: List[str],
) -> np.ndarray:
    src_feats = [_build_features(A[a]) for a in idsA]
    tgt_feats = [_build_features(B[b]) for b in idsB]
    S = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
    for i, src_feat in enumerate(src_feats):
        for j, tgt_feat in enumerate(tgt_feats):
            heuristic = _rule_score_heuristic_support(A[idsA[i]], B[idsB[j]])
            strong = _score_from_features_raw(src_feat, tgt_feat)
            S[i, j] = (0.78 * strong) + (0.10 * heuristic)
    return S


def pairwise_rule_scores(
    A: Dict[str, Dict[str, Any]],
    B: Dict[str, Dict[str, Any]],
    idsA: List[str],
    idsB: List[str],
    mode: str = "enhanced",
) -> np.ndarray:
    mode = str(mode or "enhanced").lower()
    if mode in {"heur", "heuristic", "weak"}:
        return pairwise_rule_scores_heuristic(A, B, idsA, idsB)
    return pairwise_rule_scores_enhanced(A, B, idsA, idsB)
