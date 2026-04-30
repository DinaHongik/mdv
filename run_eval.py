from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

# allow "python run_eval.py" from repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smartmap_mdv.data import build_corpus, load_pairs
from smartmap_mdv.model import MPNetEncoder, DiffCSEEncoder, DiffCLREncoder
from smartmap_mdv.constraints import combine_scores, hungarian_1to1
from smartmap_mdv.evaluate import (
    ranks_from_scores,
    hit_at_k,
    mrr,
    bootstrap_ci,
    ndcg_at_k,
    ece_from_probs,
)

BASELINES_AVAILABLE = False
TfidfEncoder = None
SbertEncoder = None
E5MultiEncoder = None
BM25Encoder = None
LogsyEncoder = None
RoBERTaDiffCSEEncoder = None
BaselineNormalizeText = None
BaselineTokenizeText = None

external_rule_scores = None
external_rule_scores_heuristic = None
external_rule_scores_enhanced = None

_BASELINES_LOAD_KEY = None
_RULE_BASELINES_LOAD_KEY = None


# ============================================================
# Helper classes / utils
# ============================================================
class ScoreWeights:
    def __init__(
        self,
        alpha_cos: float = 1.0,
        beta_type: float = 0.3,
        gamma_path: float = 0.2,
        delta_lex: float = 0.1,
    ):
        self.alpha_cos = alpha_cos
        self.beta_type = beta_type
        self.gamma_path = gamma_path
        self.delta_lex = delta_lex


def _load_module_from_path(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_baseline_imports(args=None):
    global BASELINES_AVAILABLE, TfidfEncoder, SbertEncoder, E5MultiEncoder, BM25Encoder, LogsyEncoder, RoBERTaDiffCSEEncoder, BaselineNormalizeText, BaselineTokenizeText, _BASELINES_LOAD_KEY
    baseline_module_path = getattr(args, "baseline_module_path", None) if args is not None else None
    load_key = baseline_module_path or "__default__"
    if _BASELINES_LOAD_KEY == load_key:
        return
    try:
        if baseline_module_path:
            mod = _load_module_from_path("smartmap_mdv_external_baselines", baseline_module_path)
        else:
            from smartmap_mdv import baselines as mod
        TfidfEncoder = getattr(mod, "TfidfEncoder")
        SbertEncoder = getattr(mod, "SbertEncoder")
        E5MultiEncoder = getattr(mod, "E5MultiEncoder")
        BM25Encoder = getattr(mod, "BM25Encoder")
        LogsyEncoder = getattr(mod, "LogsyEncoder")
        RoBERTaDiffCSEEncoder = getattr(mod, "RoBERTaDiffCSEEncoder")
        BaselineNormalizeText = getattr(mod, "normalize_text", None)
        BaselineTokenizeText = getattr(mod, "tokenize_text", None)
        BASELINES_AVAILABLE = True
        _BASELINES_LOAD_KEY = load_key
    except Exception as e:
        print(f"[WARN] Baseline encoders not available: {e}", file=sys.stderr)
        BASELINES_AVAILABLE = False
        TfidfEncoder = None
        SbertEncoder = None
        E5MultiEncoder = None
        BM25Encoder = None
        LogsyEncoder = None
        RoBERTaDiffCSEEncoder = None
        BaselineNormalizeText = None
        BaselineTokenizeText = None
        _BASELINES_LOAD_KEY = load_key


def ensure_rule_baseline_imports(args=None):
    global external_rule_scores, external_rule_scores_heuristic, external_rule_scores_enhanced, _RULE_BASELINES_LOAD_KEY
    rule_baselines_path = getattr(args, "rule_baselines_path", None) if args is not None else None
    load_key = rule_baselines_path or "__default__"
    if _RULE_BASELINES_LOAD_KEY == load_key:
        return
    try:
        if rule_baselines_path:
            mod = _load_module_from_path("external_rule_baselines", rule_baselines_path)
        else:
            import rule_baselines as mod
        external_rule_scores = getattr(mod, "pairwise_rule_scores", None)
        external_rule_scores_heuristic = getattr(mod, "pairwise_rule_scores_heuristic", None)
        external_rule_scores_enhanced = getattr(mod, "pairwise_rule_scores_enhanced", None)
        _RULE_BASELINES_LOAD_KEY = load_key
    except Exception:
        external_rule_scores = None
        external_rule_scores_heuristic = None
        external_rule_scores_enhanced = None
        _RULE_BASELINES_LOAD_KEY = load_key


def normalize_input_mode(input_mode: str) -> str:
    return "raw_msg" if input_mode == "msg" else input_mode


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but no GPU is available; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return device


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    z = np.sum(e, axis=1, keepdims=True)
    z = np.maximum(z, 1e-12)
    return e / z


def extract_text(item):
    if isinstance(item, tuple):
        return item[1]
    return item


def sparse_normalize_text(text: str) -> str:
    if BaselineNormalizeText is not None:
        return BaselineNormalizeText(text)
    text = str(text or "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"[\[\]\(\)\{\},:;/\\|]+", " ", text)
    text = re.sub(r"[\._$\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def sparse_tokenize_text(text: str) -> List[str]:
    if BaselineTokenizeText is not None:
        return BaselineTokenizeText(text, strong=True)
    norm = sparse_normalize_text(text)
    return [tok for tok in norm.split(" ") if tok]


def build_sparse_field_text(field: Dict[str, Any]) -> str:
    name = field_name_to_text(field)
    path = field_path_to_text(field).replace(".", " ")
    type_val = field_type_to_text(field)
    parts = [
        sparse_normalize_text(name),
        sparse_normalize_text(name),
        " ".join(sparse_tokenize_text(name)),
        sparse_normalize_text(path),
        " ".join(sparse_tokenize_text(path)),
        sparse_normalize_text(type_val),
        sparse_normalize_text(type_val),
    ]
    return " ".join(p for p in parts if p).strip()


def row_max_normalize(S: np.ndarray) -> np.ndarray:
    if S.size == 0:
        return S
    denom = np.maximum(S.max(axis=1, keepdims=True), 1e-9)
    return S / denom


def token_overlap_matrix(texts_A: List[str], texts_B: List[str]) -> np.ndarray:
    sets_A = [set(sparse_tokenize_text(t)) for t in texts_A]
    sets_B = [set(sparse_tokenize_text(t)) for t in texts_B]
    S = np.zeros((len(texts_A), len(texts_B)), dtype=np.float32)
    for i, tokens_a in enumerate(sets_A):
        if not tokens_a:
            continue
        for j, tokens_b in enumerate(sets_B):
            if not tokens_b:
                continue
            inter = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            if union > 0:
                S[i, j] = inter / union
    return S


def path_overlap_matrix(A_fields, B_fields, idsA, idsB) -> np.ndarray:
    S = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
    path_tokens_A = [
        set(sparse_tokenize_text(field_path_to_text(A_fields[aid]).replace(".", " ")))
        for aid in idsA
    ]
    path_tokens_B = [
        set(sparse_tokenize_text(field_path_to_text(B_fields[bid]).replace(".", " ")))
        for bid in idsB
    ]
    for i, tokens_a in enumerate(path_tokens_A):
        if not tokens_a:
            continue
        for j, tokens_b in enumerate(path_tokens_B):
            if not tokens_b:
                continue
            inter = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            if union > 0:
                S[i, j] = inter / union
    return S


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if ckpt_obj is None:
        return {}
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        return ckpt_obj
    if hasattr(ckpt_obj, "state_dict") and callable(ckpt_obj.state_dict):
        return ckpt_obj.state_dict()
    return {}


def safe_load_state_dict(target_model: torch.nn.Module, source_state_dict: Dict[str, torch.Tensor], strict: bool = False) -> bool:
    if not source_state_dict:
        return False
    try:
        target_model.load_state_dict(source_state_dict, strict=strict)
        return True
    except (RuntimeError, ValueError):
        target_dict = target_model.state_dict()
        filtered_dict = {
            k: v
            for k, v in source_state_dict.items()
            if k in target_dict and target_dict[k].shape == v.shape
        }
        if filtered_dict:
            target_model.load_state_dict(filtered_dict, strict=False)
            print("[WARN] Loaded state_dict with filtered layers due to shape mismatches.", file=sys.stderr)
            return True
        print("[WARN] Could not load state_dict: no matching layers found.", file=sys.stderr)
        return False


def load_run_config_from_ckpt(ckpt: str) -> Dict[str, Any]:
    if not ckpt:
        return {}
    ckpt_dir = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    if not ckpt_dir:
        return {}
    cfg_path = os.path.join(ckpt_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load run_config.json from {cfg_path}: {e}", file=sys.stderr)
        return {}


def resolve_model_config(args, ckpt: str) -> Dict[str, Any]:
    ckpt_cfg = load_run_config_from_ckpt(ckpt)
    encoder_model = args.encoder_model or ckpt_cfg.get(
        "encoder_model",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    mlm_model = args.mlm_model or ckpt_cfg.get("mlm_model", "bert-base-multilingual-cased")
    max_len = args.max_len if args.max_len is not None else int(ckpt_cfg.get("max_len", 512))
    return {"encoder_model": encoder_model, "mlm_model": mlm_model, "max_len": max_len}


def build_index_maps(A_fields, B_fields, pairs):
    a_ids = list(A_fields.keys())
    b_ids = list(B_fields.keys())
    a2i = {fid: i for i, fid in enumerate(a_ids)}
    b2j = {fid: j for j, fid in enumerate(b_ids)}

    N = len(a_ids)
    y_true_sets = [set() for _ in range(N)]
    for a, b in pairs:
        if a in a2i and b in b2j:
            y_true_sets[a2i[a]].add(b2j[b])

    y_true_idx = np.array([min(s) if len(s) > 0 else -1 for s in y_true_sets], dtype=int)
    return a_ids, b_ids, y_true_sets, y_true_idx, a2i, b2j


def filter_rows_with_gt(S, y_true_sets, y_true_idx=None):
    keep = np.array([len(s) > 0 for s in y_true_sets], dtype=bool)
    S2 = S[keep]
    y_sets2 = [y_true_sets[i] for i in np.where(keep)[0]]
    if y_true_idx is None:
        return S2, y_sets2, keep
    y_idx2 = y_true_idx[keep]
    return S2, y_sets2, y_idx2, keep


def compute_ranking_metrics(S: np.ndarray, y_true_sets):
    S2, y_sets2, _keep = filter_rows_with_gt(S, y_true_sets)
    r = ranks_from_scores(S2, y_sets2)
    return {
        "Hit@1": float(hit_at_k(r, 1)),
        "Hit@3": float(hit_at_k(r, 3)),
        "Hit@5": float(hit_at_k(r, 5)),
        "MRR": float(mrr(r)),
        "NDCG@3": float(ndcg_at_k(S2, y_sets2, k=3)),
        "NDCG@5": float(ndcg_at_k(S2, y_sets2, k=5)),
    }, S2, y_sets2


def field_type_to_text(field: Dict[str, Any]) -> str:
    t = field.get("type", "")
    if isinstance(t, dict):
        return str(t.get("base", "") or "")
    return str(t or "")


def field_name_to_text(field: Dict[str, Any]) -> str:
    return str(field.get("name", "") or "")


def field_path_to_text(field: Dict[str, Any]) -> str:
    return str(field.get("path", "") or "")


def _cosine_from_features(VA: Any, VB: Any) -> np.ndarray:
    VA = torch.from_numpy(VA.toarray() if hasattr(VA, "toarray") else VA).float()
    VB = torch.from_numpy(VB.toarray() if hasattr(VB, "toarray") else VB).float()
    VA = F.normalize(VA, dim=1)
    VB = F.normalize(VB, dim=1)
    return (VA @ VB.T).numpy()


def _exact_type_matrix(A_fields, B_fields, idsA, idsB) -> np.ndarray:
    S = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
    for i, a in enumerate(idsA):
        ta = field_type_to_text(A_fields[a]).strip().lower()
        if not ta:
            continue
        for j, b in enumerate(idsB):
            tb = field_type_to_text(B_fields[b]).strip().lower()
            if ta == tb and tb:
                S[i, j] = 1.0
    return S


def normalize_embeddings(x: Any, device: str) -> torch.Tensor:
    x = torch.as_tensor(x, device=device, dtype=torch.float32)
    return F.normalize(x, p=2, dim=1)


def encode_texts(encoder, texts: List[str], device: str) -> torch.Tensor:
    texts = [t if isinstance(t, str) and t != "" else " " for t in texts]
    with torch.no_grad():
        embs = encoder.encode(texts, device=device) if hasattr(encoder, "encode") else encoder(texts, device=device)
    return normalize_embeddings(embs, device)


def cosine_matrix_from_encoder(encoder, texts_A: List[str], texts_B: List[str], device: str) -> np.ndarray:
    VA = encode_texts(encoder, texts_A, device)
    VB = encode_texts(encoder, texts_B, device)
    return (VA @ VB.T).detach().cpu().numpy()


def instantiate_train_encoder(name: str, device: str, ckpt: str, args):
    cfg = resolve_model_config(args, ckpt)
    if name == "m":
        enc = MPNetEncoder(model_name=cfg["encoder_model"], max_length=cfg["max_len"])
    elif name == "md":
        enc = DiffCSEEncoder(
            encoder_model_name=cfg["encoder_model"],
            mlm_model_name=cfg["mlm_model"],
            max_length=cfg["max_len"],
        )
    elif name == "mdv":
        enc = DiffCLREncoder(
            encoder_model_name=cfg["encoder_model"],
            mlm_model_name=cfg["mlm_model"],
            max_length=cfg["max_len"],
        )
    else:
        raise ValueError(f"Unknown train encoder: {name}")

    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        try:
            obj = torch.load(ckpt, map_location="cpu")
        except Exception:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)

        if hasattr(obj, "encode") and hasattr(obj, "eval"):
            enc = obj
        else:
            state_dict = _extract_state_dict(obj)
            safe_load_state_dict(enc, state_dict, strict=False)

    enc.to(device).eval()
    return enc


def pairwise_rule_scores_heuristic(A_fields, B_fields, idsA, idsB, args=None):
    ensure_rule_baseline_imports(args)
    if external_rule_scores_heuristic is not None:
        return external_rule_scores_heuristic(A_fields, B_fields, idsA, idsB)
    if external_rule_scores is not None:
        return external_rule_scores(A_fields, B_fields, idsA, idsB, mode="heuristic")
    raise RuntimeError("rule_baselines.py not available")


def pairwise_rule_scores_enhanced(A_fields, B_fields, idsA, idsB, args=None):
    ensure_rule_baseline_imports(args)
    if external_rule_scores_enhanced is not None:
        return external_rule_scores_enhanced(A_fields, B_fields, idsA, idsB)
    if external_rule_scores is not None:
        return external_rule_scores(A_fields, B_fields, idsA, idsB, mode="enhanced")
    raise RuntimeError("rule_baselines.py not available")


def eval_encoder(encoder_name, A, B, txtA_map, txtB_map, idsA, idsB, device, ckpt="", args=None):
    t0 = time.time()
    name = encoder_name.lower()
    if name in {"tfidf", "sbert", "e5", "logsy", "roberta_diffcse", "bm25"}:
        ensure_baseline_imports(args)
    if name in {"rule", "rule_heur", "rule_enh"}:
        ensure_rule_baseline_imports(args)

    texts_A = [extract_text(txtA_map[i]) for i in idsA]
    texts_B = [extract_text(txtB_map[i]) for i in idsB]

    if name in {"m", "md", "mdv"}:
        enc = instantiate_train_encoder(name, device, ckpt, args)
        S = cosine_matrix_from_encoder(enc, texts_A, texts_B, device)

    elif name == "rule":
        S = pairwise_rule_scores_enhanced(A.fields, B.fields, idsA, idsB, args=args)
    elif name == "rule_heur":
        S = pairwise_rule_scores_heuristic(A.fields, B.fields, idsA, idsB, args=args)
    elif name == "rule_enh":
        S = pairwise_rule_scores_enhanced(A.fields, B.fields, idsA, idsB, args=args)

    elif name == "tfidf":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        src_sparse = [build_sparse_field_text(A.fields[aid]) for aid in idsA]
        tgt_sparse = [build_sparse_field_text(B.fields[bid]) for bid in idsB]
        enc_sparse = TfidfEncoder()
        enc_sparse.fit(src_sparse + tgt_sparse)
        S_sparse = _cosine_from_features(enc_sparse.encode(src_sparse), enc_sparse.encode(tgt_sparse))
        S_type = _exact_type_matrix(A.fields, B.fields, idsA, idsB)
        S_path = path_overlap_matrix(A.fields, B.fields, idsA, idsB)
        S_overlap = token_overlap_matrix(src_sparse, tgt_sparse)
        S = (
            0.55 * S_sparse
            + 0.15 * S_overlap
            + 0.15 * S_path
            + 0.15 * S_type
        )

    elif name == "sbert":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = SbertEncoder(device=device, max_length=args.max_len or 512)
        VA = torch.tensor(enc.encode(texts_A), dtype=torch.float32)
        VB = torch.tensor(enc.encode(texts_B), dtype=torch.float32)
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "e5":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = E5MultiEncoder(device=device, max_length=args.max_len or 512)
        VA = torch.tensor(enc.encode_queries(texts_A), dtype=torch.float32)
        VB = torch.tensor(enc.encode_passages(texts_B), dtype=torch.float32)
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "logsy":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = LogsyEncoder(
            ckpt_path=args.logsy_ckpt,
            device=device,
            max_length=args.max_len or 512,
            base_model_name=args.logsy_base_model,
        )
        VA = torch.tensor(enc.encode(texts_A), dtype=torch.float32)
        VB = torch.tensor(enc.encode(texts_B), dtype=torch.float32)
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "roberta_diffcse":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = RoBERTaDiffCSEEncoder(
            model_dir=args.roberta_diffcse_dir,
            device=device,
            max_length=args.max_len or 512,
        )
        VA = torch.tensor(enc.encode(texts_A), dtype=torch.float32)
        VB = torch.tensor(enc.encode(texts_B), dtype=torch.float32)
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "bm25":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        src_name = [field_name_to_text(A.fields[aid]) for aid in idsA]
        tgt_name = [field_name_to_text(B.fields[bid]) for bid in idsB]
        src_path = [field_path_to_text(A.fields[aid]) for aid in idsA]
        tgt_path = [field_path_to_text(B.fields[bid]) for bid in idsB]
        src_sparse = [build_sparse_field_text(A.fields[aid]) for aid in idsA]
        tgt_sparse = [build_sparse_field_text(B.fields[bid]) for bid in idsB]

        enc_full = BM25Encoder()
        enc_full.fit(tgt_sparse)
        enc_name = BM25Encoder()
        enc_name.fit(tgt_name)
        enc_path = BM25Encoder()
        enc_path.fit(tgt_path)

        S_full = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
        S_name = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
        S_path = np.zeros((len(idsA), len(idsB)), dtype=np.float32)
        for i, txt in enumerate(src_sparse):
            S_full[i] = enc_full.get_scores(txt)
        for i, txt in enumerate(src_name):
            S_name[i] = enc_name.get_scores(txt)
        for i, txt in enumerate(src_path):
            S_path[i] = enc_path.get_scores(txt)

        S_full = row_max_normalize(S_full)
        S_name = row_max_normalize(S_name)
        S_path = row_max_normalize(S_path)
        S_type = _exact_type_matrix(A.fields, B.fields, idsA, idsB)
        S_overlap = token_overlap_matrix(src_sparse, tgt_sparse)
        S_path_overlap = path_overlap_matrix(A.fields, B.fields, idsA, idsB)
        S = (
            0.40 * S_full
            + 0.20 * S_name
            + 0.10 * S_path
            + 0.15 * S_overlap
            + 0.05 * S_path_overlap
            + 0.10 * S_type
        )

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    elapsed = time.time() - t0
    return S, elapsed


def compute_component_scores(encoder, A_fields, B_fields, a_ids, b_ids, device: str):
    src_name = [field_name_to_text(A_fields[aid]) for aid in a_ids]
    tgt_name = [field_name_to_text(B_fields[bid]) for bid in b_ids]
    src_type = [field_type_to_text(A_fields[aid]) for aid in a_ids]
    tgt_type = [field_type_to_text(B_fields[bid]) for bid in b_ids]
    src_path = [field_path_to_text(A_fields[aid]) for aid in a_ids]
    tgt_path = [field_path_to_text(B_fields[bid]) for bid in b_ids]

    S_name = cosine_matrix_from_encoder(encoder, src_name, tgt_name, device)
    S_type = cosine_matrix_from_encoder(encoder, src_type, tgt_type, device)
    S_path = cosine_matrix_from_encoder(encoder, src_path, tgt_path, device)
    return S_name, S_type, S_path


def weighted_integrated_score(S_name, S_type, S_path, w_name=0.4, w_type=0.2, w_path=0.4):
    total = max(w_name + w_type + w_path, 1e-12)
    return (w_name * S_name + w_type * S_type + w_path * S_path) / total


def fit_isotonic_on_top1(probs: np.ndarray, y_true_sets):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = np.array([1.0 if (pred[i] in y_true_sets[i]) else 0.0 for i in range(len(pred))], dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(conf, acc)
    return ir


def apply_isotonic_to_probs(probs: np.ndarray, ir: IsotonicRegression):
    N, M = probs.shape
    probs_post = np.zeros_like(probs, dtype=np.float64)
    for i in range(N):
        p = probs[i]
        top1 = int(np.argmax(p))
        c0 = float(p[top1])
        c1 = float(ir.predict([c0])[0])
        rest0 = 1.0 - c0
        rest1 = max(1.0 - c1, 0.0)
        newp = np.zeros_like(p, dtype=np.float64)
        newp[top1] = c1
        if rest0 <= 1e-12:
            if M > 1 and rest1 > 0:
                per = rest1 / (M - 1)
                for j in range(M):
                    if j != top1:
                        newp[j] = per
        else:
            scale = rest1 / rest0
            for j in range(M):
                if j != top1:
                    newp[j] = float(p[j]) * scale
        newp = np.maximum(newp, 0.0)
        s = float(newp.sum())
        if s > 0:
            newp /= s
        probs_post[i] = newp
    return probs_post


def split_pairs_by_source(pairs: List[Tuple[str, str]], calib_ratio: float = 0.2, seed: int = 42):
    src_ids = sorted({a for a, _ in pairs})
    if len(src_ids) <= 1:
        return pairs, []
    rng = random.Random(seed)
    rng.shuffle(src_ids)
    n_calib = max(1, int(len(src_ids) * calib_ratio))
    n_calib = min(n_calib, len(src_ids) - 1)
    calib_src = set(src_ids[:n_calib])
    eval_pairs = [(a, b) for (a, b) in pairs if a not in calib_src]
    calib_pairs = [(a, b) for (a, b) in pairs if a in calib_src]
    return eval_pairs, calib_pairs


def prepare_eval_and_calib_pairs(args, all_pairs):
    if args.pairs_calib:
        calib_pairs = load_pairs(args.pairs_calib)
        return all_pairs, calib_pairs, "external_pairs_calib"
    if args.calibrate:
        eval_pairs, calib_pairs = split_pairs_by_source(all_pairs, calib_ratio=args.calib_ratio, seed=args.calib_seed)
        return eval_pairs, calib_pairs, "internal_source_split"
    return all_pairs, None, "none"


def maybe_apply_calibration(S_full: np.ndarray, y_true_sets_eval, y_true_sets_calib):
    if y_true_sets_calib is None:
        return None, None, None
    S_calib, y_calib, _ = filter_rows_with_gt(S_full, y_true_sets_calib)
    S_eval, y_eval, _ = filter_rows_with_gt(S_full, y_true_sets_eval)
    if S_calib.shape[0] == 0 or S_eval.shape[0] == 0:
        return None, None, None
    probs_calib = stable_softmax(S_calib)
    ir = fit_isotonic_on_top1(probs_calib, y_calib)
    probs_eval_pre = stable_softmax(S_eval)
    probs_eval_post = apply_isotonic_to_probs(probs_eval_pre, ir)
    return ir, probs_eval_pre, probs_eval_post


def sync_if_needed(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


_TYPE_DRIFT_MAP = {
    "ip": "string",
    "port": "integer",
    "integer": "string",
    "float": "integer",
    "datetime": "string",
    "boolean": "string",
    "string": "datetime",
}


def _drift_path_value(path_text: str) -> str:
    parts = [p for p in str(path_text or "").split(".") if p]
    if not parts:
        return str(path_text or "")
    return ".".join(f"slot{i}" for i in range(len(parts)))


def _drift_type_value(type_text: str) -> str:
    t = str(type_text or "").strip().lower()
    return _TYPE_DRIFT_MAP.get(t, "string" if t else "")


def _apply_text_perturbation(text_value: str, path_drift: bool = False, type_drift: bool = False) -> str:
    text_value = str(text_value or "")
    if path_drift:
        text_value = re.sub(
            r"(\[PATH\]\s+)([^\[]+)",
            lambda m: m.group(1) + _drift_path_value(m.group(2).strip()),
            text_value,
        )
    if type_drift:
        text_value = re.sub(
            r"(\[TYPE\]\s+)([^\[]+)",
            lambda m: m.group(1) + _drift_type_value(m.group(2).strip()),
            text_value,
        )
    return text_value


def apply_text_perturbations(corpus, path_drift: bool = False, type_drift: bool = False):
    return [
        (fid, _apply_text_perturbation(text, path_drift=path_drift, type_drift=type_drift))
        for fid, text in corpus
    ]


def compute_dataset_stats(A_fields, B_fields, pairs):
    a_ids, b_ids, y_true_sets, y_true_idx, a2i, b2j = build_index_maps(A_fields, B_fields, pairs)
    source_degrees = [len(s) for s in y_true_sets if len(s) > 0]

    target_to_sources: Dict[int, set] = {}
    for i, targets in enumerate(y_true_sets):
        for t in targets:
            target_to_sources.setdefault(t, set()).add(i)

    target_degrees = [len(v) for v in target_to_sources.values()]

    return {
        "n_source_fields": len(a_ids),
        "n_target_fields": len(b_ids),
        "n_gold_pairs": len(pairs),
        "n_sources_with_gt": int(sum(1 for s in y_true_sets if len(s) > 0)),
        "n_unique_targets_in_pairs": len(target_to_sources),
        "avg_targets_per_source": round(float(np.mean(source_degrees)) if source_degrees else 0.0, 4),
        "max_targets_per_source": int(max(source_degrees) if source_degrees else 0),
        "avg_sources_per_target": round(float(np.mean(target_degrees)) if target_degrees else 0.0, 4),
        "max_sources_per_target": int(max(target_degrees) if target_degrees else 0),
        "n_targets_with_multiple_sources": int(sum(1 for d in target_degrees if d > 1)),
    }


def run_stress_eval(args, device: str):
    # Shortcut robustness: perturb one signal at a time.
    presets = [
        ("Original", dict()),
        ("MaskName", dict(mask_name=True)),
        ("MaskType", dict(drop_type=True)),
        ("MaskPath", dict(drop_path=True)),
        ("PathDrift", dict(path_drift=True)),
        ("TypeDrift", dict(type_drift=True)),
    ]

    out = {}
    for label, overrides in presets:
        A2, B2, corpusA2, corpusB2 = build_corpus(
            args.fieldsA,
            args.fieldsB,
            mask_name=overrides.get("mask_name", False),
            input_mode=args.input_mode,
            drop_type=overrides.get("drop_type", False),
            drop_path=overrides.get("drop_path", False),
            drop_desc=overrides.get("drop_desc", False),
            drop_example=overrides.get("drop_example", False),
            no_placeholder=overrides.get("no_placeholder", args.no_placeholder),
        )
        if overrides.get("path_drift") or overrides.get("type_drift"):
            corpusA2 = apply_text_perturbations(
                corpusA2,
                path_drift=overrides.get("path_drift", False),
                type_drift=overrides.get("type_drift", False),
            )
        txtA_map2 = dict(corpusA2)
        txtB_map2 = dict(corpusB2)
        eval_pairs, calib_pairs, calib_source = prepare_eval_and_calib_pairs(args, load_pairs(args.pairs))
        a_ids2, b_ids2, y_true_sets_eval2, y_true_idx_eval2, _, _ = build_index_maps(A2.fields, B2.fields, eval_pairs)
        S2, secs2 = eval_encoder(args.encoder, A2, B2, txtA_map2, txtB_map2, a_ids2, b_ids2, device, ckpt=args.ckpt, args=args)
        metrics2, _, _ = compute_ranking_metrics(S2, y_true_sets_eval2)
        metrics2["timing_sec"] = round(float(secs2), 3)
        out[label] = {k: round(float(v), 4) for k, v in metrics2.items()}
    return out


def run_latency_eval(args, A, B, txtA_map, txtB_map, a_ids, b_ids, device: str):
    name = args.encoder.lower()
    texts_A = [extract_text(txtA_map[i]) for i in a_ids]
    texts_B = [extract_text(txtB_map[i]) for i in b_ids]

    out = {
        "encoder": args.encoder,
        "input_mode": args.input_mode,
        "n_source_fields": len(a_ids),
        "n_target_fields": len(b_ids),
    }

    if name in {"m", "md", "mdv"}:
        encoder = instantiate_train_encoder(name, device, args.ckpt, args)

        sync_if_needed(device)
        t0 = time.perf_counter()
        VA = encode_texts(encoder, texts_A, device)
        sync_if_needed(device)
        encode_a_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        VB = encode_texts(encoder, texts_B, device)
        sync_if_needed(device)
        encode_b_sec = time.perf_counter() - t1

        t2 = time.perf_counter()
        S = (VA @ VB.T).detach().cpu().numpy()
        sync_if_needed(device)
        score_sec = time.perf_counter() - t2

        total_sec = encode_a_sec + encode_b_sec + score_sec
        out.update({
            "encode_A_sec": round(float(encode_a_sec), 4),
            "encode_B_sec": round(float(encode_b_sec), 4),
            "score_sec": round(float(score_sec), 4),
            "total_sec": round(float(total_sec), 4),
            "ms_per_source": round(float(1000.0 * total_sec / max(len(a_ids), 1)), 4),
        })

        if args.latency_with_components:
            t3 = time.perf_counter()
            S_name, S_type, S_path = compute_component_scores(encoder, A.fields, B.fields, a_ids, b_ids, device)
            S_weighted = weighted_integrated_score(
                S_name, S_type, S_path,
                w_name=args.s_name_weight,
                w_type=args.s_type_weight,
                w_path=args.s_path_weight,
            )
            sync_if_needed(device)
            comp_sec = time.perf_counter() - t3
            out["integrated_S_sec"] = round(float(comp_sec), 4)
            out["integrated_total_sec"] = round(float(total_sec + comp_sec), 4)
            out["integrated_ms_per_source"] = round(float(1000.0 * (total_sec + comp_sec) / max(len(a_ids), 1)), 4)

    else:
        t0 = time.perf_counter()
        S, secs = eval_encoder(args.encoder, A, B, txtA_map, txtB_map, a_ids, b_ids, device, ckpt=args.ckpt, args=args)
        sync_if_needed(device)
        total_sec = time.perf_counter() - t0
        out.update({
            "total_sec": round(float(total_sec), 4),
            "ms_per_source": round(float(1000.0 * total_sec / max(len(a_ids), 1)), 4),
        })

    return out


def run_latency_breakdown(args, A, B, txtA_map, txtB_map, a_ids, b_ids, y_true_sets_eval, device: str):
    """Measure three-stage latency: embedding -> scoring/ranking -> calibration."""
    name = args.encoder.lower()
    if name not in {"m", "md", "mdv"}:
        raise ValueError("latency3 supports encoders {m, md, mdv}")

    texts_A = [extract_text(txtA_map[i]) for i in a_ids]
    texts_B = [extract_text(txtB_map[i]) for i in b_ids]

    encoder = instantiate_train_encoder(name, device, args.ckpt, args)

    # 1) Embedding
    sync_if_needed(device)
    t0 = time.perf_counter()
    VA = encode_texts(encoder, texts_A, device)
    VB = encode_texts(encoder, texts_B, device)
    sync_if_needed(device)
    embed_sec = time.perf_counter() - t0

    # 2) Similarity scoring (cosine matmul)
    t1 = time.perf_counter()
    S_full = (VA @ VB.T).detach().cpu().numpy()
    sync_if_needed(device)
    score_sec = time.perf_counter() - t1

    # 3) Calibration prep (softmax on valid rows)
    S_eval, _y_sets2, _ = filter_rows_with_gt(S_full, y_true_sets_eval)
    t2 = time.perf_counter()
    _ = stable_softmax(S_eval)
    calib_sec = time.perf_counter() - t2

    total_sec = embed_sec + score_sec + calib_sec
    ms_per_source = 1000.0 * total_sec / max(len(a_ids), 1)
    return {
        "encoder": args.encoder,
        "input_mode": args.input_mode,
        "n_source_fields": len(a_ids),
        "n_target_fields": len(b_ids),
        "embed_sec": round(float(embed_sec), 4),
        "score_rank_sec": round(float(score_sec), 4),
        "calibration_sec": round(float(calib_sec), 4),
        "total_sec": round(float(total_sec), 4),
        "ms_per_source": round(float(ms_per_source), 4),
    }


# ============================================================
# Main
# ============================================================
def main(args):
    args.input_mode = normalize_input_mode(args.input_mode)
    device = resolve_device(args.device)

    A, B, corpusA, corpusB = build_corpus(
        args.fieldsA,
        args.fieldsB,
        mask_name=args.mask_name,
        input_mode=args.input_mode,
        drop_type=args.drop_type,
        drop_path=args.drop_path,
        drop_desc=args.drop_desc,
        drop_example=args.drop_example,
        no_placeholder=args.no_placeholder,
    )

    txtA_map = dict(corpusA)
    txtB_map = dict(corpusB)

    all_pairs = load_pairs(args.pairs)
    eval_pairs, calib_pairs, calib_source = prepare_eval_and_calib_pairs(args, all_pairs)
    ranking_pairs = all_pairs if args.mode == "table4" else eval_pairs

    a_ids, b_ids, y_true_sets_eval, y_true_idx_eval, _a2i, _b2j = build_index_maps(A.fields, B.fields, ranking_pairs)
    _a_ids2, _b_ids2, y_true_sets_calib, y_true_idx_calib, _a2i2, _b2j2 = build_index_maps(A.fields, B.fields, calib_pairs or [])

    if args.mode == "dataset_stats":
        stats = compute_dataset_stats(A.fields, B.fields, all_pairs)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.mode == "stress":
        out = run_stress_eval(args, device)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.mode == "latency":
        out = run_latency_eval(args, A, B, txtA_map, txtB_map, a_ids, b_ids, device)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.mode == "latency3":
        out = run_latency_breakdown(args, A, B, txtA_map, txtB_map, a_ids, b_ids, y_true_sets_eval, device)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # --------------------------------------------------------
    # Table 3: component-wise integrated score
    # --------------------------------------------------------
    if args.mode == "table3":
        if args.encoder not in {"m", "md", "mdv"}:
            print("table3 requires encoder in {m, md, mdv}", file=sys.stderr)
            return
        encoder = instantiate_train_encoder(args.encoder, device, args.ckpt, args)
        model_tag = args.encoder.upper()
        print("Computing component similarities (name/type/path)...", file=sys.stderr)
        S_name, S_type, S_path = compute_component_scores(encoder, A.fields, B.fields, a_ids, b_ids, device)
        S_weighted = weighted_integrated_score(
            S_name, S_type, S_path,
            w_name=args.s_name_weight,
            w_type=args.s_type_weight,
            w_path=args.s_path_weight,
        )

        results = {}
        for label, S_comp in [
            ("Name-only", S_name),
            ("Type-only", S_type),
            ("Path-only", S_path),
            ("S (weighted)", S_weighted),
        ]:
            metrics, _S2, _y2 = compute_ranking_metrics(S_comp, y_true_sets_eval)
            results[label] = {k: round(v, 4) for k, v in metrics.items()}

        print(json.dumps({model_tag: results}, ensure_ascii=False, indent=2))
        return

    # --------------------------------------------------------
    # Table 4: calibration on integrated score
    # --------------------------------------------------------
    if args.mode == "table4":
        if args.encoder not in {"m", "md", "mdv"}:
            print("table4 requires encoder in {m, md, mdv}", file=sys.stderr)
            return

        encoder = instantiate_train_encoder(args.encoder, device, args.ckpt, args)
        model_tag = args.encoder.upper()
        print("Computing integrated score S for calibration/evaluation...", file=sys.stderr)
        S_name, S_type, S_path = compute_component_scores(encoder, A.fields, B.fields, a_ids, b_ids, device)
        S = weighted_integrated_score(
            S_name, S_type, S_path,
            w_name=args.s_name_weight,
            w_type=args.s_type_weight,
            w_path=args.s_path_weight,
        )

        metrics, S_eval, y_eval = compute_ranking_metrics(S, y_true_sets_eval)
        probs_pre = stable_softmax(S_eval)
        ECE_pre = ece_from_probs(probs_pre, y_eval, n_bins=args.ece_bins)

        ECE_post = ECE_pre
        calibration_used = "none"
        if args.calibrate:
            ir, probs_eval_pre, probs_eval_post = maybe_apply_calibration(
                S_full=S,
                y_true_sets_eval=y_true_sets_eval,
                y_true_sets_calib=y_true_sets_calib if calib_pairs else None,
            )
            if ir is not None and probs_eval_post is not None:
                ECE_post = ece_from_probs(probs_eval_post, y_eval, n_bins=args.ece_bins)
                calibration_used = f"isotonic({calib_source})"
            else:
                calibration_used = "isotonic_failed_no_valid_calib_rows"

        out = {
            "encoder": args.encoder,
            "mode": "integrated_score_S",
            "input_mode": args.input_mode,
            "calibration": calibration_used,
            "calibration_source": calib_source,
            "eval_pairs": len(ranking_pairs),
            "calib_pairs": len(calib_pairs or []),
            "Hit@1": round(metrics["Hit@1"], 4),
            "Hit@3": round(metrics["Hit@3"], 4),
            "Hit@5": round(metrics["Hit@5"], 4),
            "MRR": round(metrics["MRR"], 4),
            "NDCG@3": round(metrics["NDCG@3"], 4),
            "NDCG@5": round(metrics["NDCG@5"], 4),
            "ECE_pre": round(float(ECE_pre), 4),
            "ECE_post": round(float(ECE_post), 4),
            "s_name_weight": args.s_name_weight,
            "s_type_weight": args.s_type_weight,
            "s_path_weight": args.s_path_weight,
        }
        print(json.dumps({model_tag: out}, ensure_ascii=False, indent=2))
        return

    # --------------------------------------------------------
    # Mode: all (main encoder + baselines)
    # --------------------------------------------------------
    if args.mode == "all":
        baseline_list = [args.encoder, "rule_heur", "rule_enh", "tfidf", "sbert", "logsy", "roberta_diffcse", "e5", "bm25"]
        seen = set()
        baseline_list = [x for x in baseline_list if not (x in seen or seen.add(x))]

        results = {}
        for enc_name in baseline_list:
            try:
                print(f"Running {enc_name}...", file=sys.stderr)
                current_ckpt = args.ckpt if enc_name in {"m", "md", "mdv"} else ""
                S, secs_enc = eval_encoder(enc_name, A, B, txtA_map, txtB_map, a_ids, b_ids, device, ckpt=current_ckpt, args=args)
                metrics, _S2, _y2 = compute_ranking_metrics(S, y_true_sets_eval)
                if enc_name == args.encoder:
                    key = args.main_label
                elif enc_name == "rule_heur":
                    key = "RULE_HEUR"
                elif enc_name == "rule_enh":
                    key = "RULE_ENH"
                else:
                    key = enc_name.upper()
                results[key] = {
                    "Hit@1": round(metrics["Hit@1"], 4),
                    "Hit@3": round(metrics["Hit@3"], 4),
                    "Hit@5": round(metrics["Hit@5"], 4),
                    "MRR": round(metrics["MRR"], 4),
                    "NDCG@3": round(metrics["NDCG@3"], 4),
                    "NDCG@5": round(metrics["NDCG@5"], 4),
                    "timing_sec": round(float(secs_enc), 3),
                }
                print(f"{enc_name}: Hit@1={metrics['Hit@1']:.3f}", file=sys.stderr)
            except Exception as e:
                print(f"{enc_name} failed: {e}", file=sys.stderr)

        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # --------------------------------------------------------
    # Default / single evaluation
    # --------------------------------------------------------
    S, secs_enc = eval_encoder(
        args.encoder,
        A, B,
        txtA_map, txtB_map,
        a_ids, b_ids,
        device,
        ckpt=args.ckpt,
        args=args,
    )

    if args.use_constraints:
        w = ScoreWeights(
            alpha_cos=args.alpha,
            beta_type=args.beta,
            gamma_path=args.gamma,
            delta_lex=args.delta,
        )
        S, _parts = combine_scores(S, A.fields, B.fields, a_ids, b_ids, w)

    metrics, S_eval, y_eval = compute_ranking_metrics(S, y_true_sets_eval)
    probs_pre = stable_softmax(S_eval)
    ECE_pre = ece_from_probs(probs_pre, y_eval, n_bins=args.ece_bins)
    ECE_post = ECE_pre
    probs_post = None
    calibration_used = "none"

    if args.calibrate:
        ir, probs_eval_pre, probs_eval_post = maybe_apply_calibration(
            S_full=S,
            y_true_sets_eval=y_true_sets_eval,
            y_true_sets_calib=y_true_sets_calib if calib_pairs else None,
        )
        if ir is not None and probs_eval_post is not None:
            probs_post = probs_eval_post
            ECE_post = ece_from_probs(probs_post, y_eval, n_bins=args.ece_bins)
            calibration_used = f"isotonic({calib_source})"
        else:
            calibration_used = "isotonic_failed_no_valid_calib_rows"

    assign_acc, coverage = None, None
    if args.hungarian:
        S_for_assign = S_eval
        if probs_post is not None:
            S_for_assign = np.log(np.clip(probs_post, 1e-12, 1.0))
        r_idx, c_idx = hungarian_1to1(S_for_assign, tau=args.assign_tau)
        correct = sum(1 for i, j in zip(r_idx, c_idx) if (j >= 0 and j in y_eval[i]))
        assigned = sum(1 for _i, j in zip(r_idx, c_idx) if j >= 0)
        assign_acc = correct / assigned if assigned > 0 else 0.0
        coverage = assigned / len(y_eval) if len(y_eval) > 0 else 0.0

    out_ci = None
    if args.bootstrap > 0:
        lo, hi = bootstrap_ci((ranks_from_scores(S_eval, y_eval) == 1).astype(float), n_boot=args.bootstrap)
        out_ci = [round(float(lo), 4), round(float(hi), 4)]

    out = {
        "encoder": args.encoder,
        "ckpt": args.ckpt,
        "input_mode": args.input_mode,
        "use_constraints": args.use_constraints,
        "hungarian": args.hungarian,
        "calibration_method": calibration_used,
        "calibration_source": calib_source,
        "eval_pairs": len(eval_pairs),
        "calib_pairs": len(calib_pairs or []),
        "timing_sec": round(float(secs_enc), 3),
        "Hit@1": round(metrics["Hit@1"], 4),
        "Hit@3": round(metrics["Hit@3"], 4),
        "Hit@5": round(metrics["Hit@5"], 4),
        "MRR": round(metrics["MRR"], 4),
        "NDCG@3": round(metrics["NDCG@3"], 4),
        "NDCG@5": round(metrics["NDCG@5"], 4),
        "ECE_pre": round(float(ECE_pre), 4),
        "ECE_post": round(float(ECE_post), 4),
    }
    if assign_acc is not None:
        out["AssignAcc_1to1"] = round(float(assign_acc), 4)
        out["Coverage"] = round(float(coverage), 4)
    if out_ci is not None:
        out["Hit@1_CI95"] = out_ci

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Core inputs
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--fieldsA", required=True)
    ap.add_argument("--fieldsB", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs_calib", default="", help="Optional labeled pair file for calibration only.")

    # Encoder / mode
    ap.add_argument(
        "--encoder",
        choices=["m", "md", "mdv", "rule", "rule_heur", "rule_enh", "tfidf", "sbert", "logsy", "roberta_diffcse", "e5", "bm25"],
        default="mdv",
    )
    ap.add_argument(
        "--mode",
        choices=["single", "all", "table3", "table4", "stress", "latency", "latency3", "dataset_stats"],
        default="single",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Input serialization
    ap.add_argument("--input_mode", choices=["raw_msg", "flat_field", "nmo", "msg"], default="nmo")
    ap.add_argument("--mask_name", action="store_true")
    ap.add_argument("--drop_type", action="store_true")
    ap.add_argument("--drop_path", action="store_true")
    ap.add_argument("--drop_desc", action="store_true")
    ap.add_argument("--drop_example", action="store_true")
    ap.add_argument("--no_placeholder", action="store_true")

    # Model config overrides (optional; otherwise inferred from run_config.json)
    ap.add_argument("--encoder_model", default=None)
    ap.add_argument("--mlm_model", default=None)
    ap.add_argument("--max_len", type=int, default=None)

    # Constraints / assignment
    ap.add_argument("--use_constraints", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0, help="Cosine weight")
    ap.add_argument("--beta", type=float, default=0.3, help="Type compatibility weight")
    ap.add_argument("--gamma", type=float, default=0.2, help="Path similarity weight")
    ap.add_argument("--delta", type=float, default=0.1, help="Lexical bonus weight")
    ap.add_argument("--hungarian", action="store_true")
    ap.add_argument("--assign_tau", type=float, default=0.5)

    # Calibration / robustness
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--calib_ratio", type=float, default=0.2)
    ap.add_argument("--calib_seed", type=int, default=42)

    # Table 3 / Table 4 integrated score weights
    ap.add_argument("--s_name_weight", type=float, default=0.4)
    ap.add_argument("--s_type_weight", type=float, default=0.2)
    ap.add_argument("--s_path_weight", type=float, default=0.4)

    # Misc
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--main_label", default="MAIN")

    # Optional baseline-specific paths
    ap.add_argument("--baseline_module_path", default=None)
    ap.add_argument("--rule_baselines_path", default=None)
    ap.add_argument("--logsy_ckpt", default=None)
    ap.add_argument("--logsy_base_model", default="bert-base-uncased")
    ap.add_argument("--roberta_diffcse_dir", default=None)

    # Latency options
    ap.add_argument("--latency_with_components", action="store_true")

    args = ap.parse_args()
    main(args)
