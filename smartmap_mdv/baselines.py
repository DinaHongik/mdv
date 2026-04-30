import os
import math
from typing import Sequence, Dict, List
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Shared tokenizer / text normalization
# ============================================================
TOKEN_ALIAS = {
    "src": ["source", "ip"],
    "dst": ["destination", "ip"],
    "sip": ["source", "ip"],
    "dip": ["destination", "ip"],
    "sourceip": ["source", "ip"],
    "destip": ["destination", "ip"],
    "destinationip": ["destination", "ip"],
    "spt": ["source", "port"],
    "dpt": ["destination", "port"],
    "sport": ["source", "port"],
    "dport": ["destination", "port"],
    "sourceport": ["source", "port"],
    "destport": ["destination", "port"],
    "destinationport": ["destination", "port"],
    "saddr": ["source", "address"],
    "daddr": ["destination", "address"],
    "smac": ["source", "mac"],
    "dmac": ["destination", "mac"],
    "srcmac": ["source", "mac"],
    "dstmac": ["destination", "mac"],
    "usr": ["user"],
    "uid": ["user", "id"],
    "sess": ["session"],
    "sessid": ["session", "id"],
    "sessionid": ["session", "id"],
    "ts": ["timestamp"],
    "msg": ["message"],
    "proto": ["protocol"],
    "svc": ["service"],
    "svcid": ["service", "id"],
    "appid": ["application", "id"],
    "ruleid": ["rule", "id"],
    "policyid": ["policy", "id"],
    "country": ["country"],
    "cc": ["country", "code"],
    "srcip": ["source", "ip"],
    "dstip": ["destination", "ip"],
    "srcipcc": ["source", "ip", "country", "code"],
    "sipcc": ["source", "ip", "country", "code"],
    "dipcc": ["destination", "ip", "country", "code"],
    "natsrc": ["source", "nat", "ip"],
    "natdst": ["destination", "nat", "ip"],
    "natsport": ["source", "nat", "port"],
    "natdport": ["destination", "nat", "port"],
    "profileid": ["profile", "id"],
    "module_name": ["module", "name"],
    "moduleflag": ["module", "flag"],
    "ap_protocol": ["application", "protocol"],
    "descrption": ["description"],
}

KO_CANONICAL = {
    "출발지": "source",
    "목적지": "destination",
    "원본": "source",
    "원격지": "destination",
    "클라이언트": "source",
    "서버": "destination",
    "주소": "address",
    "아이피": "ip",
    "ip주소": "ip address",
    "포트": "port",
    "프로토콜": "protocol",
    "세션": "session",
    "시간": "time",
    "시각": "time",
    "시작": "start",
    "종료": "end",
    "이름": "name",
    "사용자": "user",
    "모듈": "module",
    "설명": "description",
    "행위": "action",
    "동작": "action",
    "국가": "country",
    "코드": "code",
    "서비스": "service",
    "버전": "version",
    "메시지": "message",
    "로그": "log",
    "규칙": "rule",
    "정책": "policy",
    "응답": "response",
    "요청": "request",
    "탐지": "detect",
    "공격": "attack",
    "위험": "threat",
    "백신": "antivirus",
    "방화벽": "firewall",
    "웹방화벽": "waf",
    "아이디": "id",
}


KO_CANONICAL.update({
    "로그": "log",
    "로그인": "login",
    "방식": "method",
    "전달가능": "forwardable",
    "인증": "authentication",
    "인증티켓": "authentication ticket",
    "여부": "is",
    "목적지": "destination",
    "출발지": "source",
    "서비스명": "service name",
    "서비스": "service",
    "외부": "external",
    "거래": "transaction",
    "네트워크": "network",
    "속도": "speed",
    "프로토콜": "protocol",
    "호스트명": "host name",
    "호스트": "host",
    "클라이언트": "client",
    "서버": "server",
    "프로세스": "process",
    "부모": "parent",
    "시스템": "system",
    "대상": "target",
    "사용자": "user",
    "사용자명": "username",
    "계정": "account",
    "장비": "device",
    "도메인": "domain",
    "참여": "joined",
    "주소": "address",
    "이름": "name",
    "파일명": "filename",
    "파일": "file",
    "이벤트": "event",
    "스레드": "thread",
    "레코드": "record",
    "포트": "port",
    "유형": "type",
    "타입": "type",
    "한정자": "qualifier",
    "레지스트리": "registry",
    "키": "key",
    "악성": "malicious",
    "운영체제": "operating system",
    "아키텍처": "architecture",
    "세션": "session",
    "경과시간": "elapsed time",
    "시간": "time",
    "초": "sec",
    "대화형": "interactive",
    "명령어": "command",
    "오류": "error",
    "메시지": "message",
    "심각도": "severity",
    "정책": "policy",
    "응답": "response",
    "요청": "request",
    "본문": "body",
    "국가": "country",
    "코드": "code",
    "접속": "connection",
    "권한": "privilege",
    "상승": "escalation",
    "버전": "version",
    "길이": "length",
    "용량": "capacity",
    "상태": "status",
    "설명": "description",
    "명": "name",
    "공격자": "attacker",
    "위협": "threat",
    "공격": "attack",
})


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Preserve case boundaries before lowercasing.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = text.replace("`", " ")
    text = re.sub(r"[\[\]\(\)\{\},:;/\\|]+", " ", text)
    text = re.sub(r"[\._$\-]+", " ", text)
    for ko, en in sorted(KO_CANONICAL.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(ko, f" {en} ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def default_tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    base_tokens = [t for t in text.split(" ") if t]
    toks: List[str] = []
    for tok in base_tokens:
        toks.append(tok)

        # Recover common syslog abbreviations into semantically richer tokens.
        if tok in TOKEN_ALIAS:
            toks.extend(TOKEN_ALIAS[tok])

        # Split compact mixed tokens like sourceip, sessionid, ruleid.
        for suffix, expansion in [
            ("ip", ["ip"]),
            ("port", ["port"]),
            ("id", ["id"]),
            ("name", ["name"]),
            ("code", ["code"]),
            ("time", ["time"]),
            ("mac", ["mac"]),
        ]:
            if tok not in TOKEN_ALIAS and tok.endswith(suffix) and len(tok) > len(suffix) + 1:
                head = tok[: -len(suffix)]
                toks.append(head)
                toks.extend(expansion)

        if tok.startswith("src") and tok not in {"src"}:
            toks.extend(["source", tok[3:]])
        if tok.startswith("dst") and tok not in {"dst"}:
            toks.extend(["destination", tok[3:]])
        if tok.startswith("sip") and tok not in {"sip"}:
            toks.extend(["source", "ip", tok[3:]])
        if tok.startswith("dip") and tok not in {"dip"}:
            toks.extend(["destination", "ip", tok[3:]])
        if tok.startswith("nat"):
            toks.append("nat")
            rem = tok[3:]
            if rem:
                toks.append(rem)
                if rem in TOKEN_ALIAS:
                    toks.extend(TOKEN_ALIAS[rem])

        synonym_map = {
            "dest": "destination",
            "remote": "destination",
            "target": "destination",
            "origin": "source",
            "client": "source",
            "server": "destination",
            "usr": "user",
            "msg": "message",
            "proto": "protocol",
            "svc": "service",
        }
        if tok in synonym_map:
            toks.append(synonym_map[tok])

        # Add compact canonical variants to bridge surface-form mismatch.
        if tok in {"source", "destination", "session", "protocol", "message", "address"}:
            toks.append(tok[:3] if len(tok) > 3 else tok)
        if tok == "source":
            toks.extend(["src"])
        elif tok == "destination":
            toks.extend(["dst", "dest"])
        elif tok == "protocol":
            toks.extend(["proto"])
        elif tok == "message":
            toks.extend(["msg"])
        elif tok == "service":
            toks.extend(["svc"])
        elif tok == "session":
            toks.extend(["sess"])
        elif tok == "user":
            toks.extend(["usr"])

        # Add short character shingles for compact identifiers.
        if 4 <= len(tok) <= 20 and re.fullmatch(r"[a-z0-9]+", tok):
            for n in (3, 4):
                if len(tok) >= n:
                    toks.extend(tok[i:i+n] for i in range(len(tok) - n + 1))

    return toks


def tokenize_text(text: str, strong: bool = True) -> List[str]:
    toks = default_tokenize(text)
    if strong:
        return toks
    return [t for t in normalize_text(text).split(" ") if t]


# ============================================================
# TF-IDF Encoder
# ============================================================
class TfidfEncoder:
    """Hybrid character/word TF-IDF encoder for schema-style field names."""
    def __init__(self, **kwargs):
        self.char_vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="char_wb",
            ngram_range=(2, 6),
            max_df=0.95,
            min_df=1,
            sublinear_tf=True,
            **kwargs,
        )
        self.word_vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="word",
            tokenizer=default_tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.98,
            min_df=1,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus: Sequence[str]):
        corpus = [c if isinstance(c, str) else "" for c in corpus]
        self.char_vectorizer.fit(corpus)
        self.word_vectorizer.fit(corpus)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]):
        if not self._fitted:
            raise RuntimeError("TfidfEncoder: fit() must be called first.")
        texts = [t if isinstance(t, str) else "" for t in texts]
        char_feats = self.char_vectorizer.transform(texts)
        word_feats = self.word_vectorizer.transform(texts)
        return hstack([char_feats, word_feats])


# ============================================================
# BM25 Encoder
# ============================================================
class BM25Encoder:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.idf = {}
        self.avgdl = 0.0
        self.N = 0

    def _tokenize(self, text: str):
        toks = default_tokenize(text)
        norm = normalize_text(text)
        compact = norm.replace(" ", "")
        if 4 <= len(compact) <= 64 and re.fullmatch(r"[a-z0-9]+", compact):
            for n in (3, 4, 5):
                if len(compact) >= n:
                    toks.extend(compact[i:i+n] for i in range(len(compact) - n + 1))
        return toks

    def fit(self, docs: Sequence[str]):
        self.docs = [self._tokenize(d) for d in docs]
        self.N = len(self.docs)
        if self.N == 0:
            self.avgdl = 0.0
            self.idf = {}
            return self

        doc_lens = [len(d) for d in self.docs]
        self.avgdl = float(sum(doc_lens) / len(doc_lens))

        df: Dict[str, int] = {}
        for d in self.docs:
            seen = set()
            for w in d:
                if w not in seen:
                    df[w] = df.get(w, 0) + 1
                    seen.add(w)

        self.idf = {
            w: math.log(1 + (self.N - df_w + 0.5) / (df_w + 0.5))
            for w, df_w in df.items()
        }
        return self

    def get_scores(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        if self.N == 0:
            return np.zeros(0, dtype=np.float32)

        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.docs):
            doc_len = len(doc)
            if doc_len == 0:
                continue

            tf: Dict[str, int] = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1

            score = 0.0
            denom = self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-9))
            for w in q_tokens:
                if w not in tf:
                    continue
                idf = self.idf.get(w, 0.0)
                f = tf[w]
                score += idf * (f * (self.k1 + 1)) / (f + denom)

            scores[i] = score
        return scores


# ============================================================
# Base dense encoder helpers
# ============================================================
class DenseEncoderBase:
    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _normalize(self, emb):
        return F.normalize(emb, p=2, dim=1)


# ============================================================
# SBERT Encoder
# ============================================================
class SbertEncoder(DenseEncoderBase):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [normalize_text(t) if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                emb = self._normalize(emb)
                all_embs.append(emb.cpu())

        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# E5 Encoder
# ============================================================
class E5MultiEncoder(DenseEncoderBase):
    """intfloat/multilingual-e5-base encoder"""
    def __init__(self, model_name="intfloat/multilingual-e5-base", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def _prepare(self, texts, prefix="passage"):
        return [f"{prefix}: {normalize_text(t)}" if isinstance(t, str) else f"{prefix}: " for t in texts]

    def encode_queries(self, texts, batch_size=32):
        return self._encode_with_prefix(texts, prefix="query", batch_size=batch_size)

    def encode_passages(self, texts, batch_size=32):
        return self._encode_with_prefix(texts, prefix="passage", batch_size=batch_size)

    def _encode_with_prefix(self, texts, prefix="passage", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        texts = self._prepare(texts, prefix=prefix)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                emb = self._normalize(emb)
                all_embs.append(emb.cpu())

        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()

    def encode(self, texts, batch_size=32):
        return self._encode_with_prefix(texts, prefix="passage", batch_size=batch_size)


# ============================================================
# Logsy Encoder
# ============================================================
class LogsyEncoder(DenseEncoderBase):
    """Logsy checkpoint loader"""
    def __init__(self, ckpt_path=None, device=None, max_length=512, base_model_name="bert-base-multilingual-cased"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModel.from_pretrained(base_model_name).to(self.device)

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict):
                    if "state_dict" in state:
                        self.model.load_state_dict(state["state_dict"], strict=False)
                    elif "model_state_dict" in state:
                        self.model.load_state_dict(state["model_state_dict"], strict=False)
                    else:
                        self.model.load_state_dict(state, strict=False)
            except Exception:
                pass

        self.model.eval()
        self.max_length = max_length

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [normalize_text(t) if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                emb = self._normalize(emb)
                all_embs.append(emb.cpu())

        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# RoBERTa DiffCSE Encoder
# ============================================================
class RoBERTaDiffCSEEncoder(DenseEncoderBase):
    """DiffCSE RoBERTa checkpoint loader"""
    def __init__(self, model_dir=None, device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        if model_dir is None:
            model_dir = "diffcse_roberta_baseline_20251007/checkpoint-1500"

        if not os.path.isdir(model_dir):
            raise ValueError(f"DiffCSE RoBERTa no directory: {model_dir}")

        print(f"Loading RoBERTa-DiffCSE from: {model_dir}")

        from transformers import AutoTokenizer, RobertaModel, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        self.model = RobertaModel(config).to(self.device)

        ckpt_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(ckpt_path):
            try:
                import warnings
                import io
                warnings.filterwarnings("ignore")

                with open(ckpt_path, "rb") as f:
                    buffer = io.BytesIO(f.read())
                    state_dict = torch.load(buffer, map_location=self.device, weights_only=False)

                if isinstance(state_dict, dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace("roberta.", "") if k.startswith("roberta.") else k
                        new_state_dict[new_key] = v

                    missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                    print("Loaded checkpoint successfully")
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e})")
        else:
            print(f"Warning: pytorch_model.bin not found at {ckpt_path}")

        self.model.eval()

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [normalize_text(t) if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                emb = self._normalize(emb)
                all_embs.append(emb.cpu())

        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()
