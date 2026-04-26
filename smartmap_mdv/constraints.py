import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import difflib

TYPE_COMPAT = {
    ("ip", "ip"): 1.0, ("port", "port"): 1.0, ("integer", "integer"): 1.0, ("string", "string"): 1.0,
    ("integer", "port"): 0.6, ("port", "integer"): 0.6, ("string", "datetime"): 0.2, ("datetime", "string"): 0.2,
}

def type_compat(a_rec, b_rec):
    ta = a_rec.get("type", {}).get("base", "")
    tb = b_rec.get("type", {}).get("base", "")
    return TYPE_COMPAT.get((ta, tb), 0.0)

def path_similarity(a_rec, b_rec):
    def toks(p):
        p = p or ""
        return set(re.split(r"[\./_\$\[\]:]+", p.lower()))
    
    sa = toks(a_rec.get("path", ""))
    sb = toks(b_rec.get("path", ""))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def lexical_bonus(a_rec, b_rec):
    na = (a_rec.get("name", "") or "").lower().replace("_", "")
    nb = (b_rec.get("name", "") or "").lower().replace("_", "")
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()

def combine_scores(cos, A, B, idsA, idsB, weights):
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
    
    S = (weights.alpha_cos * cos +
         weights.beta_type * T +
         weights.gamma_path * P +
         weights.delta_lex * L)
    
    return S, {"cos": cos, "type": T, "path": P, "lex": L}

def hungarian_1to1(S, tau=None):

    S = np.asarray(S, dtype=float)
    N, M = S.shape

    if tau is not None:
        dummy = np.full((N, 1), float(tau))
        S_aug = np.concatenate([S, dummy], axis=1)
        M_aug = M + 1
    else:
        S_aug = S
        M_aug = M

    S_aug = np.clip(S_aug, 0.0, 1.0)
    C = 1.0 - S_aug

    r_all, c_all = linear_sum_assignment(C)

    r_sel, c_sel = [], []
    for i, j in zip(r_all, c_all):
        if tau is not None and j == M:           
            r_sel.append(i); c_sel.append(-1)
        else:
            if tau is not None and S[i, j] < tau:
                r_sel.append(i); c_sel.append(-1)
            else:
                r_sel.append(i); c_sel.append(j)

    return np.asarray(r_sel), np.asarray(c_sel)
