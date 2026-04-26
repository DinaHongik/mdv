import os
import math
from typing import List, Sequence, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================
# TF-IDF Encoder
# ============================================================
class TfidfEncoder:
    """ TF-IDF encoder"""
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            **kwargs,
        )
        self._fitted = False

    def fit(self, corpus: Sequence[str]):
        corpus = [c if isinstance(c, str) else "" for c in corpus]
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]):
        if not self._fitted:
            raise RuntimeError("TfidfEncoder: fit().")
        texts = [t if isinstance(t, str) else "" for t in texts]
        return self.vectorizer.transform(texts)


# ============================================================
# BM25 Encoder
# ============================================================
class BM25Encoder:
    """ BM25 """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return text.lower().split()

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

        self.idf = {w: math.log(1 + (self.N - df_w + 0.5) / (df_w + 0.5)) for w, df_w in df.items()}
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
# SBERT Encoder
# ============================================================
class SbertEncoder:
    """SBERT (SentenceTransformer) encoder"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, max_length=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                all_embs.append(emb.cpu())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# E5 Encoder
# ============================================================
class E5MultiEncoder:
    """intfloat/multilingual-e5-base encoder"""
    def __init__(self, model_name="intfloat/multilingual-e5-base", device=None, max_length=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                all_embs.append(emb.cpu())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# Logsy Encoder
# ============================================================
class LogsyEncoder:
    """Logsy checkpoint loader"""
    def __init__(self, ckpt_path=None, device=None, max_length=128, base_model_name="bert-base-uncased"):
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

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                all_embs.append(emb.cpu())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# RoBERTa DiffCSE Encoder
# ============================================================
class RoBERTaDiffCSEEncoder:
    """ DiffCSE RoBERTa checkpoint loader"""
    def __init__(self, model_dir=None, device=None, max_length=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        if model_dir is None:
            model_dir = "diffcse_roberta_baseline_20251007/checkpoint-1500"
        
        if not os.path.isdir(model_dir):
            raise ValueError(f"DiffCSE RoBERTa no directory: {model_dir}")
        
        print(f"Loading RoBERTa-DiffCSE from: {model_dir}")
        
        from transformers import AutoTokenizer, RobertaModel, AutoConfig
        
        # tokenizer, config load
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        
        self.model = RobertaModel(config).to(self.device)
        
        # state_dict load
        ckpt_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(ckpt_path):
            try:
                import warnings
                warnings.filterwarnings('ignore')
                
                import io
                with open(ckpt_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    state_dict = torch.load(buffer, map_location=self.device, weights_only=False)
                
                if isinstance(state_dict, dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace('roberta.', '') if k.startswith('roberta.') else k
                        new_state_dict[new_key] = v
                    
                    missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"Loaded checkpoint successfully")
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
                else:
                    print(f"Warning: state_dict is not a dict: {type(state_dict)}")
                    
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e})")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: pytorch_model.bin not found at {ckpt_path}")
        
        self.model.eval()

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
                all_embs.append(emb.cpu())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return torch.cat(all_embs, dim=0).numpy()
