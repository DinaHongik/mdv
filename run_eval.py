import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression

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

try:
    from smartmap_mdv.baselines import (
        TfidfEncoder,
        SbertEncoder,
        E5MultiEncoder,
        BM25Encoder,
        LogsyEncoder,
        RoBERTaDiffCSEEncoder
    )
    BASELINES_AVAILABLE = True
except Exception as e:
    print(f"Warning: Baselines not available: {e}", file=sys.stderr)
    BASELINES_AVAILABLE = False


class ScoreWeights:
    def __init__(self, alpha_cos=1.0, beta_name=0.0, gamma_type=0.0, delta_path=0.0):
        self.alpha_cos = alpha_cos
        self.beta_name = beta_name
        self.gamma_type = gamma_type
        self.delta_path = delta_path


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


def extract_text(item):
    if isinstance(item, tuple):
        return item[1]
    return item


def eval_encoder(encoder_name, A, B, txtA, txtB, idsA, idsB, device, ckpt=""):
    t0 = time.time()
    name = encoder_name.lower()

    if name in ["m", "md", "mdv"]:
        from collections import OrderedDict

        if name == "m":
            enc = MPNetEncoder()
        elif name == "md":
            enc = DiffCSEEncoder()
        else:
            enc = DiffCLREncoder()

        if ckpt and os.path.exists(ckpt):
            obj = None
            try:
                obj = torch.load(ckpt, map_location="cpu")
            except Exception:
                obj = None

            if obj is None:
                try:
                    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}", file=sys.stderr)
                    obj = None

            if obj is not None:
                if hasattr(obj, "to") and hasattr(obj, "eval") and hasattr(obj, "encode"):
                    enc = obj
                    print(f"Loaded full encoder object from {ckpt}", file=sys.stderr)
                elif isinstance(obj, (dict, OrderedDict)):
                    sd = obj
                    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                        sd = sd["state_dict"]
                    if isinstance(sd, dict) and "encoder" in sd and isinstance(sd["encoder"], (dict, OrderedDict)):
                        sd = sd["encoder"]
                    missing, unexpected = enc.load_state_dict(sd, strict=False)
                    print(
                        f"Loaded state_dict from {ckpt} (missing={len(missing)}, unexpected={len(unexpected)})",
                        file=sys.stderr,
                    )

        enc.to(device).eval()

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        with torch.no_grad():
            VA = enc.encode(texts_A, device=device)
            VB = enc.encode(texts_B, device=device)

        VA = F.normalize(torch.as_tensor(VA).to(device), dim=1)
        VB = F.normalize(torch.as_tensor(VB).to(device), dim=1)
        S = (VA @ VB.T).cpu().numpy()

    elif name == "tfidf":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = TfidfEncoder()

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        corpus = texts_A + texts_B
        enc.fit(corpus)
        VA = enc.encode(texts_A)
        VB = enc.encode(texts_B)

        VA = torch.from_numpy(VA.toarray() if hasattr(VA, "toarray") else VA).float()
        VB = torch.from_numpy(VB.toarray() if hasattr(VB, "toarray") else VB).float()
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "sbert":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = SbertEncoder(device=device)

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        VA = torch.tensor(enc.encode(texts_A)).float()
        VB = torch.tensor(enc.encode(texts_B)).float()
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "e5":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = E5MultiEncoder(device=device)

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        VA = torch.tensor(enc.encode(texts_A)).float()
        VB = torch.tensor(enc.encode(texts_B)).float()
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "logsy":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = LogsyEncoder(device=device)

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        VA = torch.tensor(enc.encode(texts_A)).float()
        VB = torch.tensor(enc.encode(texts_B)).float()
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "roberta_diffcse":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = RoBERTaDiffCSEEncoder(device=device)

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        VA = torch.tensor(enc.encode(texts_A)).float()
        VB = torch.tensor(enc.encode(texts_B)).float()
        VA = F.normalize(VA, dim=1)
        VB = F.normalize(VB, dim=1)
        S = (VA @ VB.T).numpy()

    elif name == "bm25":
        if not BASELINES_AVAILABLE:
            raise ImportError("Baselines not available")
        enc = BM25Encoder()

        texts_A = [extract_text(txtA[i]) for i in idsA]
        texts_B = [extract_text(txtB[i]) for i in idsB]

        enc.fit(texts_B)
        S = np.zeros((len(idsA), len(idsB)))
        for i, txt in enumerate(texts_A):
            S[i] = enc.get_scores(txt)

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    elapsed = time.time() - t0
    return S, elapsed



def evaluate_component(encoder, src_texts, tgt_texts, device, y_true_sets, top_ks=[1, 3, 5], ndcg_ks=[3, 5]):
    """Table 3"""
    with torch.no_grad():
        VA = encoder.encode(src_texts, device=device)
        VB = encoder.encode(tgt_texts, device=device)

    VA = F.normalize(torch.as_tensor(VA).to(device), dim=1)
    VB = F.normalize(torch.as_tensor(VB).to(device), dim=1)
    S = (VA @ VB.T).cpu().numpy()

    # GT row
    S2, y_sets2, _keep = filter_rows_with_gt(S, y_true_sets)
    r = ranks_from_scores(S2, y_sets2)

    topk_results = {k: hit_at_k(r, k) for k in top_ks}
    ndcg_results = {k: ndcg_at_k(S2, y_sets2, k=k) for k in ndcg_ks}
    mrr_score = mrr(r)

    return {
        "Hit@1": float(topk_results[1]),
        "Hit@3": float(topk_results[3]),
        "Hit@5": float(topk_results[5]),
        "NDCG@3": float(ndcg_results[3]),
        "NDCG@5": float(ndcg_results[5]),
        "MRR": float(mrr_score),
    }, S


def fit_isotonic_on_top1(probs: np.ndarray, y_true_sets):
    """
     isotonic:
    - x: top-1 confidence
    - y: top-1 correctness(= pred ∈ GTset)
    """
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

        # 나머지 합
        rest0 = 1.0 - c0
        rest1 = 1.0 - c1
        if rest1 < 0.0:
            rest1 = 0.0

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


def main(args):
    A, B, txtA, txtB = build_corpus(
        args.fieldsA, args.fieldsB,
        mask_name=args.mask_name,
        input_mode=getattr(args, "input_mode", "nmo"),
        drop_type=getattr(args, "drop_type", False),
        drop_path=getattr(args, "drop_path", False),
        no_placeholder=getattr(args, "no_placeholder", False)
    )
    pairs = load_pairs(args.pairs)
    A_fields = A.fields if hasattr(A, "fields") else A
    B_fields = B.fields if hasattr(B, "fields") else B

    a_ids, b_ids, y_true_sets, y_true_idx, a2i, b2j = build_index_maps(A_fields, B_fields, pairs)
    idsA = list(range(len(a_ids)))
    idsB = list(range(len(b_ids)))

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    ckpt = getattr(args, "ckpt", "")

    results = {}

    # -------------------------
    # Mode: table3 (Ablation)
    # -------------------------
    if args.mode == "table3":
        print("\n[Table 3] Ablation Study", file=sys.stderr)

        if args.encoder != "mdv":
            print("Warning: table3 mode requires --encoder mdv", file=sys.stderr)
            return

        encoder = DiffCLREncoder()
        if ckpt and os.path.exists(ckpt):
            try:
                state_dict = torch.load(ckpt, map_location=device)
                if isinstance(state_dict, dict):
                    encoder.load_state_dict(state_dict, strict=False)
                else:
                    encoder = state_dict
            except Exception as e:
                print(f"Failed to load checkpoint: {e}", file=sys.stderr)

        encoder.to(device).eval()

        # Name-only
        src_name = [str(A_fields[aid].get("name", "")) for aid in a_ids]
        tgt_name = [str(B_fields[bid].get("name", "")) for bid in b_ids]
        results["Name-only"], S_name = evaluate_component(encoder, src_name, tgt_name, device, y_true_sets)

        # Type-only
        src_type = [str(A_fields[aid].get("type", "")) for aid in a_ids]
        tgt_type = [str(B_fields[bid].get("type", "")) for bid in b_ids]
        results["Type-only"], S_type = evaluate_component(encoder, src_type, tgt_type, device, y_true_sets)

        # Path-only
        src_path = [str(A_fields[aid].get("path", "")) for aid in a_ids]
        tgt_path = [str(B_fields[bid].get("path", "")) for bid in b_ids]
        results["Path-only"], S_path = evaluate_component(encoder, src_path, tgt_path, device, y_true_sets)

        # Weighted combination
        S_weighted = 0.4 * S_name + 0.2 * S_type + 0.4 * S_path

        S2, y_sets2, _keep = filter_rows_with_gt(S_weighted, y_true_sets)
        r = ranks_from_scores(S2, y_sets2)

        results["S (weighted)"] = {
            "Hit@1": float(hit_at_k(r, 1)),
            "Hit@3": float(hit_at_k(r, 3)),
            "Hit@5": float(hit_at_k(r, 5)),
            "NDCG@3": float(ndcg_at_k(S2, y_sets2, k=3)),
            "NDCG@5": float(ndcg_at_k(S2, y_sets2, k=5)),
            "MRR": float(mrr(r)),
        }

        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # -------------------------
    # Mode: table4 (S + ECE)
    # -------------------------
    if args.mode == "table4":
        print("\n[Table 4] Integrated Score + Isotonic Calibration", file=sys.stderr)

        if args.encoder != "mdv":
            print("Warning: table4 mode requires --encoder mdv", file=sys.stderr)
            return

        encoder = DiffCLREncoder()
        if ckpt and os.path.exists(ckpt):
            try:
                state_dict = torch.load(ckpt, map_location=device)
                if isinstance(state_dict, dict):
                    encoder.load_state_dict(state_dict, strict=False)
                else:
                    encoder = state_dict
            except Exception as e:
                print(f"Failed to load checkpoint: {e}", file=sys.stderr)

        encoder.to(device).eval()

        src_name = [str(A_fields[aid].get("name", "")) for aid in a_ids]
        tgt_name = [str(B_fields[bid].get("name", "")) for bid in b_ids]
        src_type = [str(A_fields[aid].get("type", "")) for aid in a_ids]
        tgt_type = [str(B_fields[bid].get("type", "")) for bid in b_ids]
        src_path = [str(A_fields[aid].get("path", "")) for aid in a_ids]
        tgt_path = [str(B_fields[bid].get("path", "")) for bid in b_ids]

        print("Computing component similarities (name/type/path)...", file=sys.stderr)
        with torch.no_grad():
            VA_name = encoder.encode(src_name, device=device)
            VB_name = encoder.encode(tgt_name, device=device)
            VA_name = F.normalize(torch.as_tensor(VA_name).to(device), dim=1)
            VB_name = F.normalize(torch.as_tensor(VB_name).to(device), dim=1)
            S_name = (VA_name @ VB_name.T).cpu().numpy()

            VA_type = encoder.encode(src_type, device=device)
            VB_type = encoder.encode(tgt_type, device=device)
            VA_type = F.normalize(torch.as_tensor(VA_type).to(device), dim=1)
            VB_type = F.normalize(torch.as_tensor(VB_type).to(device), dim=1)
            S_type = (VA_type @ VB_type.T).cpu().numpy()

            VA_path = encoder.encode(src_path, device=device)
            VB_path = encoder.encode(tgt_path, device=device)
            VA_path = F.normalize(torch.as_tensor(VA_path).to(device), dim=1)
            VB_path = F.normalize(torch.as_tensor(VB_path).to(device), dim=1)
            S_path = (VA_path @ VB_path.T).cpu().numpy()

        S = 0.4 * S_name + 0.2 * S_type + 0.4 * S_path
        print("Integrated Score S weights: name=0.4, type=0.2, path=0.4", file=sys.stderr)

        S2, y_sets2, _keep = filter_rows_with_gt(S, y_true_sets)

        # Ranking metrics (GT rows)
        r = ranks_from_scores(S2, y_sets2)
        hit1 = hit_at_k(r, 1)
        hit3 = hit_at_k(r, 3)
        hit5 = hit_at_k(r, 5)
        MRR = mrr(r)
        ndcg3 = ndcg_at_k(S2, y_sets2, k=3)
        ndcg5 = ndcg_at_k(S2, y_sets2, k=5)

        # ECE (Before)
        probs_pre = softmax(S2, axis=1)
        ECE_pre = ece_from_probs(probs_pre, y_sets2, n_bins=15)

        # Isotonic calibration (option)
        ECE_post = ECE_pre
        if getattr(args, "calibrate", False):
            print("[Calibration] Isotonic on top-1 confidence (GT rows only)", file=sys.stderr)
            ir = fit_isotonic_on_top1(probs_pre, y_sets2)
            probs_post = apply_isotonic_to_probs(probs_pre, ir)
            ECE_post = ece_from_probs(probs_post, y_sets2, n_bins=15)
            print(f"[Calibration] ECE: {ECE_pre:.4f} -> {ECE_post:.4f}", file=sys.stderr)

        out = {
            "encoder": "mdv",
            "mode": "integrated_score_S",
            "calibration": "isotonic" if getattr(args, "calibrate", False) else "none",
            "Hit@1": round(float(hit1), 4),
            "Hit@3": round(float(hit3), 4),
            "Hit@5": round(float(hit5), 4),
            "MRR": round(float(MRR), 4),
            "NDCG@3": round(float(ndcg3), 4),
            "NDCG@5": round(float(ndcg5), 4),
            "ECE_pre": round(float(ECE_pre), 4),
            "ECE_post": round(float(ECE_post), 4),
        }

        print(json.dumps({"MDV": out}, ensure_ascii=False, indent=2))
        return

    # -------------------------
    # Mode: all (baselines)
    # -------------------------
    if args.mode == "all":
        baseline_list = ["mdv", "tfidf", "sbert", "logsy", "roberta_diffcse", "e5", "bm25"]

        for enc_name in baseline_list:
            try:
                print(f"Running {enc_name}...", file=sys.stderr)
                current_ckpt = ckpt if enc_name == args.encoder else ""

                S, secs_enc = eval_encoder(enc_name, A, B, txtA, txtB, idsA, idsB, device, ckpt=current_ckpt)

                S2, y_sets2, _keep = filter_rows_with_gt(S, y_true_sets)
                r = ranks_from_scores(S2, y_sets2)

                hit1 = hit_at_k(r, 1)
                hit3 = hit_at_k(r, 3)
                hit5 = hit_at_k(r, 5)
                MRR = mrr(r)
                ndcg3 = ndcg_at_k(S2, y_sets2, k=3)
                ndcg5 = ndcg_at_k(S2, y_sets2, k=5)

                results[enc_name.upper() if enc_name != "mdv" else "MDV"] = {
                    "Hit@1": round(float(hit1), 4),
                    "Hit@3": round(float(hit3), 4),
                    "Hit@5": round(float(hit5), 4),
                    "MRR": round(float(MRR), 4),
                    "NDCG@3": round(float(ndcg3), 4),
                    "NDCG@5": round(float(ndcg5), 4),
                }
                print(f"{enc_name}: Hit@1={hit1:.3f}", file=sys.stderr)
            except Exception as e:
                print(f"{enc_name} failed: {e}", file=sys.stderr)

        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # -------------------------
    # Default: mdv (single)
    # -------------------------
    S, secs_enc = eval_encoder(args.encoder, A, B, txtA, txtB, idsA, idsB, device, ckpt=ckpt)

    # Constraints integration (option)
    if args.use_constraints:
        w = ScoreWeights(
            alpha_cos=args.alpha,
            beta_name=args.beta,
            gamma_type=args.gamma,
            delta_path=args.delta
        )
        S, _parts = combine_scores(S, A.fields, B.fields, a_ids, b_ids, w)

    S2, y_sets2, y_idx2, _keep = filter_rows_with_gt(S, y_true_sets, y_true_idx)

    # Ranking metrics
    r = ranks_from_scores(S2, y_sets2)
    hit1 = hit_at_k(r, 1)
    hit3 = hit_at_k(r, 3)
    hit5 = hit_at_k(r, 5)
    MRR = mrr(r)
    ndcg3 = ndcg_at_k(S2, y_sets2, k=3)
    ndcg5 = ndcg_at_k(S2, y_sets2, k=5)

    # ECE (Before)
    probs_pre = softmax(S2, axis=1)
    ECE_pre = ece_from_probs(probs_pre, y_sets2, n_bins=15)

    # Isotonic  (option)
    ECE_post = ECE_pre
    probs_post = None
    if getattr(args, "calibrate", False):
        print("[Calibration] Isotonic on top-1 confidence (GT rows only)", file=sys.stderr)
        ir = fit_isotonic_on_top1(probs_pre, y_sets2)
        probs_post = apply_isotonic_to_probs(probs_pre, ir)
        ECE_post = ece_from_probs(probs_post, y_sets2, n_bins=15)
        print(f"[Calibration] ECE: {ECE_pre:.4f} -> {ECE_post:.4f}", file=sys.stderr)

    # hungarian (option)
    assign_acc, coverage = None, None
    if args.hungarian:
        S_for_assign = S2
        if probs_post is not None:
            S_for_assign = np.log(np.clip(probs_post, 1e-12, 1.0))

        r_idx, c_idx = hungarian_1to1(S_for_assign, tau=getattr(args, "assign_tau", 0.5))
        correct = sum(1 for i, j in zip(r_idx, c_idx) if j in y_sets2[i])
        assign_acc = correct / len(r_idx) if len(r_idx) > 0 else 0.0
        coverage = len(r_idx) / len(y_sets2) if len(y_sets2) > 0 else 0.0

    # bootstrap CI (option)
    out_ci = None
    if args.bootstrap > 0:
        lo, hi = bootstrap_ci((r == 1).astype(float), n_boot=args.bootstrap)
        out_ci = [round(lo, 4), round(hi, 4)]

    out = {
        "encoder": args.encoder,
        "ckpt": ckpt,
        "use_constraints": args.use_constraints,
        "hungarian": args.hungarian,
        "calibration_method": "isotonic" if getattr(args, "calibrate", False) else "none",
        "input_mode": getattr(args, "input_mode", "nmo"),
        "timing_sec": round(float(secs_enc), 3),
        "Hit@1": round(float(hit1), 4),
        "Hit@3": round(float(hit3), 4),
        "Hit@5": round(float(hit5), 4),
        "MRR": round(float(MRR), 4),
        "NDCG@3": round(float(ndcg3), 4),
        "NDCG@5": round(float(ndcg5), 4),
        "ECE_pre": round(float(ECE_pre), 4),
        "ECE_post": round(float(ECE_post), 4),
    }

    if assign_acc is not None:
        out["AssignAcc_1to1"] = round(float(assign_acc), 4)
        out["Coverage"] = round(float(coverage), 4)
    if out_ci is not None:
        out["Hit@1_CI95"] = out_ci

    if args.mode == "mdv":
        print(json.dumps({"MDV": out}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--fieldsA", required=True)
    ap.add_argument("--fieldsB", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--encoder", choices=["m", "md", "mdv", "tfidf", "sbert", "logsy", "roberta_diffcse", "e5", "bm25"], default="mdv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["mdv", "all", "table3", "table4"], default="mdv")
    ap.add_argument("--use_constraints", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--hungarian", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--mask_name", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--assign_tau", type=float, default=0.5)
    ap.add_argument("--input_mode", choices=["nmo", "msg"], default="nmo")
    ap.add_argument("--drop_type", action="store_true")
    ap.add_argument("--drop_path", action="store_true")
    ap.add_argument("--no_placeholder", action="store_true")
    args = ap.parse_args()
    main(args)
