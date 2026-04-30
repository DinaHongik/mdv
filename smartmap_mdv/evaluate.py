from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def ranks_from_scores(S: np.ndarray, y_true_idx_or_set):
    if S.size == 0:
        return np.array([], dtype=int)
        
    order = np.argsort(-S, axis=1)
    N = S.shape[0]
    ranks = np.empty(N, dtype=int)
    for i in range(N):
        truth = y_true_idx_or_set[i]
        if isinstance(truth, (set, list, tuple)):
            r_min = 10**9
            for tj in truth:
                pos = int(np.where(order[i] == tj)[0][0]) + 1
                if pos < r_min:
                    r_min = pos
            ranks[i] = r_min
        else:
            ranks[i] = int(np.where(order[i] == truth)[0][0]) + 1
    return ranks


def hit_at_k(ranks: np.ndarray, k: int = 1) -> float:
    return float((ranks <= k).mean())


def mrr(ranks: np.ndarray) -> float:
    return float((1.0 / ranks).mean())


def ndcg_at_k(S: np.ndarray, y_true_sets, k: int = 5) -> float:
    """NDCG@k """
    N = S.shape[0]
    ndcg_scores = []

    for i in range(N):
        sorted_indices = np.argsort(-S[i])

        dcg = 0.0
        for j in range(min(k, len(sorted_indices))):
            idx = sorted_indices[j]
            if idx in y_true_sets[i]:
                dcg += 1.0 / np.log2(j + 2)

        idcg = 0.0
        num_relevant = len(y_true_sets[i])
        for j in range(min(k, num_relevant)):
            idcg += 1.0 / np.log2(j + 2)

        ndcg_scores.append((dcg / idcg) if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores))


def mean_top1_margin(S: np.ndarray) -> float:
    part = np.partition(-S, kth=1, axis=1)
    top1 = -part[:, 0]
    top2 = -part[:, 1]
    return float(np.mean(top1 - top2))


# -----------------------------
# Calibration / ECE utilities
# -----------------------------
def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """overflow-safe softmax"""
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    z = np.sum(e, axis=1, keepdims=True)
    z = np.maximum(z, 1e-12)
    return e / z


def _looks_like_probs(X: np.ndarray, tol_sum: float = 1e-3) -> bool:
    if X.ndim != 2:
        return False
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return False
    if X.min() < -1e-6 or X.max() > 1.0 + 1e-6:
        return False
    row_sum = X.sum(axis=1)
    return bool(np.all(np.abs(row_sum - 1.0) < tol_sum))


def _acc_from_pred_and_truth(pred: np.ndarray, truth) -> np.ndarray:

    # scalar truth
    if np.isscalar(truth):
        truth = np.array([truth])

    # multi-GT: list of sets
    if isinstance(truth, (list, tuple)) and len(truth) > 0 and isinstance(truth[0], (set, list, tuple)):
        acc = np.zeros(len(pred), dtype=float)
        for i in range(len(pred)):
            gt = truth[i]
            # if gt is empty, acc=0 
            if gt is None or len(gt) == 0:
                acc[i] = 0.0
            else:
                acc[i] = 1.0 if (pred[i] in gt) else 0.0
        return acc

    # single-GT: ndarray / list of int
    truth_arr = np.asarray(truth)
    return (pred == truth_arr).astype(float)


def ece_from_scores(
    S: np.ndarray,
    y_true_idx_or_set,
    n_bins: int = 15,
    tau: float = 1.0
) -> float:
    # 1) detect probs or score
    if _looks_like_probs(S):
        probs = S
    else:
        tau = float(tau)
        if tau <= 0:
            raise ValueError("tau must be > 0")
        probs = _stable_softmax(S / tau)

    # 2) top-1 confidence / prediction
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    # 3) accuracy (single-GT or multi-GT)
    acc = _acc_from_pred_and_truth(pred, y_true_idx_or_set)

    # 4) ECE binning
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def ece_from_probs(probs: np.ndarray, y_true_idx_or_set, n_bins: int = 15) -> float:

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = _acc_from_pred_and_truth(pred, y_true_idx_or_set)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def ece_from_top1_conf(conf: np.ndarray, acc: np.ndarray, n_bins: int = 15) -> float:

    conf = np.asarray(conf, dtype=float)
    acc = np.asarray(acc, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def assign_accuracy_1to1(S: np.ndarray, y_true_idx: np.ndarray) -> float:
    row_ind, col_ind = linear_sum_assignment(-S)
    assigned = np.empty_like(y_true_idx)
    assigned[row_ind] = col_ind
    return float((assigned == y_true_idx).mean())


def candidate_stats(S: np.ndarray) -> dict:
    M = S.shape[1]
    return {"avg_candidates": float(M), "min_candidates": int(M), "max_candidates": int(M)}


def compute_all_metrics(S: np.ndarray, y_true_idx_or_set, n_bins: int = 15, tau: float = 1.0):
    ranks = ranks_from_scores(S, y_true_idx_or_set)
    return {
        "NDCG@3": ndcg_at_k(S, y_true_idx_or_set if isinstance(y_true_idx_or_set[0], (set, list, tuple)) else [{y} for y in y_true_idx_or_set], k=3),
        "NDCG@5": ndcg_at_k(S, y_true_idx_or_set if isinstance(y_true_idx_or_set[0], (set, list, tuple)) else [{y} for y in y_true_idx_or_set], k=5),
        "Hit@1": hit_at_k(ranks, 1),
        "Hit@3": hit_at_k(ranks, 3),
        "Hit@5": hit_at_k(ranks, 5),
        "MRR": mrr(ranks),
        "ECE": ece_from_scores(S, y_true_idx_or_set, n_bins=n_bins, tau=tau),
        "AssignAcc_1to1": assign_accuracy_1to1(S, np.asarray(y_true_idx_or_set) if not isinstance(y_true_idx_or_set[0], (set, list, tuple)) else np.zeros(S.shape[0], dtype=int)),
        "Top1Margin": mean_top1_margin(S),
        **candidate_stats(S),
    }


def randomized_hit1_baseline(S: np.ndarray, y_true_idx: np.ndarray) -> float:
    M = S.shape[1]
    perm = np.random.permutation(M)
    S_rand = S[:, perm]
    y_rand = np.array([int(np.where(perm == j)[0][0]) for j in y_true_idx])
    ranks = ranks_from_scores(S_rand, y_rand)
    return hit_at_k(ranks, 1)


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    boot_means = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


class Evaluator:
    def __init__(self, scores: np.ndarray, y_true_idx_or_set):
        self.S = scores
        self.y_true = y_true_idx_or_set
        self.ranks = ranks_from_scores(self.S, self.y_true)

    def get_hitatk(self, k: int):
        return hit_at_k(self.ranks, k)

    def get_mrr(self):
        return mrr(self.ranks)

    def get_ece(self, n_bins: int = 15, tau: float = 1.0):
        return ece_from_scores(self.S, self.y_true, n_bins=n_bins, tau=tau)

    def get_assignment_accuracy(self, y_true_sets: dict[str, set[str]], a_ids: list[str], b_ids: list[str]):
        b_map = {b: idx for idx, b in enumerate(b_ids)}
        true_idx = []
        for a in a_ids:
            targets = y_true_sets.get(a, set())
            if len(targets) >= 1:
                true_idx.append(b_map[next(iter(targets))])
            else:
                true_idx.append(-1)
        true_idx = np.array(true_idx)
        return assign_accuracy_1to1(self.S, true_idx)

    def get_bootstrap_ci(self, metric: str, n_boot: int = 1000):
        if metric.lower() in ["hitat1", "hit@1"]:
            data = (self.ranks <= 1).astype(int)
            return bootstrap_ci(data, n_boot)
        elif metric.lower() in ["mrr"]:
            data = 1.0 / self.ranks
            return bootstrap_ci(data, n_boot)
        elif metric.lower() in ["hitat3", "hit@3"]:
            data = (self.ranks <= 3).astype(int)
            return bootstrap_ci(data, n_boot)
        elif metric.lower() in ["hitat5", "hit@5"]:
            data = (self.ranks <= 5).astype(int)
            return bootstrap_ci(data, n_boot)
        else:
            raise NotImplementedError(f"Bootstrap CI for metric {metric} is not implemented.")
