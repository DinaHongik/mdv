import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import difflib

TYPE_COMPAT = {
    ("ip", "ip"): 1.0,
    ("port", "port"): 1.0,
    ("integer", "integer"): 1.0,
    ("string", "string"): 1.0,
    ("datetime", "datetime"): 1.0,
    ("integer", "port"): 0.6,
    ("port", "integer"): 0.6,
    ("string", "datetime"): 0.2,
    ("datetime", "string"): 0.2,
}


def _type_base(x):
    # Normalize the type field into a lowercase base-type string.
    if isinstance(x, dict):
        return str(x.get("base", "")).strip().lower()
    return str(x or "").strip().lower()


def type_compat(a_rec, b_rec):
    # Look up a coarse compatibility score between two field types.
    ta = _type_base(a_rec.get("type", ""))
    tb = _type_base(b_rec.get("type", ""))
    return TYPE_COMPAT.get((ta, tb), 0.0)


def _path_tokens(p):
    # Split a schema path into comparable lowercase tokens.
    p = str(p or "").lower()
    toks = re.split(r"[\./_\$\[\]:]+", p)
    return {t for t in toks if t}


def path_similarity(a_rec, b_rec):
    # Use Jaccard overlap over path tokens as a lightweight structural signal.
    sa = _path_tokens(a_rec.get("path", ""))
    sb = _path_tokens(b_rec.get("path", ""))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_bonus(a_rec, b_rec):
    # Compare normalized field names with a character-level similarity ratio.
    na = re.sub(r"[_\W]+", "", str(a_rec.get("name", "") or "").lower())
    nb = re.sub(r"[_\W]+", "", str(b_rec.get("name", "") or "").lower())
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def combine_scores(cos, A, B, idsA, idsB, weights):
    # Build auxiliary pairwise signals, then combine them with cosine similarity.
    N, M = cos.shape
    T = np.zeros((N, M), dtype=float)
    P = np.zeros((N, M), dtype=float)
    L = np.zeros((N, M), dtype=float)

    for i, a in enumerate(idsA):
        ar = A[a]
        for j, b in enumerate(idsB):
            br = B[b]
            T[i, j] = type_compat(ar, br)
            P[i, j] = path_similarity(ar, br)
            L[i, j] = lexical_bonus(ar, br)

    raw = (
        weights.alpha_cos * cos
        + weights.beta_type * T
        + weights.gamma_path * P
        + weights.delta_lex * L
    )

    denom = (
        weights.alpha_cos
        + weights.beta_type
        + weights.gamma_path
        + weights.delta_lex
    )
    S = raw / max(denom, 1e-9)

    return S, {"cos": cos, "type": T, "path": P, "lex": L}


def hungarian_1to1(S, tau=None):
    """
    Run 1-to-1 assignment with optional unmatched dummies.

    If tau is set, each source row gets one dummy column so the solver can
    leave the row unmatched when all real matches are below the threshold.
    """
    S = np.asarray(S, dtype=float)
    N, M = S.shape

    if tau is not None:
        # Add one dummy target per source row with score tau.
        dummy = np.full((N, N), float(tau), dtype=float)
        S_aug = np.concatenate([S, dummy], axis=1)
    else:
        S_aug = S

    # Hungarian solves a minimum-cost problem, so convert scores to costs.
    C = 1.0 - S_aug
    row_ind, col_ind = linear_sum_assignment(C)

    out_rows, out_cols = [], []
    for i, j in zip(row_ind, col_ind):
        if tau is not None and j >= M:
            # Assigned to a dummy column: mark this source as unmatched.
            out_rows.append(i)
            out_cols.append(-1)
        else:
            if tau is not None and S[i, j] < tau:
                # Reject low-scoring real matches even if the solver picked them.
                out_rows.append(i)
                out_cols.append(-1)
            else:
                out_rows.append(i)
                out_cols.append(j)

    return np.asarray(out_rows), np.asarray(out_cols)
