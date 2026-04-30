"""
This script handles the self-supervised training of the M, MD, and MDV models
for semantic schema mapping.

Input Modes:
- raw_msg   : Treat each line as raw context text.
- flat_field: Serialize each CSV line into flattened field text (without tags).
- nmo       : Serialize each CSV line into NMO format ([NAME]...[TYPE]...).
- msg       : Backward-compatible alias for raw_msg.
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from smartmap_mdv.data import to_nmo_string
from smartmap_mdv.losses import nt_xent, varclr_regularizer
from smartmap_mdv.model import DiffCLREncoder, DiffCSEEncoder, MPNetEncoder


# ------------------------------------------------------------
# Self-supervised training script for M / MD / MDV variants
# ------------------------------------------------------------


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "1", "y"}:
        return True
    if v.lower() in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int, deterministic: bool = True):
    # Seed every relevant RNG so training runs are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    # Accept several checkpoint layouts and extract a plain state_dict.
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


def safe_load_state_dict(
    target_model: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> bool:
    # Try strict loading first, then fall back to shape-matched parameters only.
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
            print("[WARN] Loaded state_dict with filtered layers due to shape mismatches.")
            return True
        print("[WARN] Could not load state_dict: no matching layers found.")
        return False


class CsvLogDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        input_mode: str = "raw_msg",
        encoding: str = "utf-8",
        multiplier: int = 1,
        mask_name: bool = False,
        drop_type: bool = False,
        drop_path: bool = False,
        drop_desc: bool = False,
        drop_example: bool = False,
        no_placeholder: bool = False,
    ):
        # This dataset reads line-based CSV/text inputs and converts them into
        # the text format expected by the selected training input mode.
        self.input_mode = "raw_msg" if input_mode == "msg" else input_mode
        self.mask_name = mask_name
        self.drop_type = drop_type
        self.drop_path = drop_path
        self.drop_desc = drop_desc
        self.drop_example = drop_example
        self.no_placeholder = no_placeholder

        self.rows: List[Any] = []
        self.header_fields: List[str] = []

        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                self.rows = self._load_rows(f)
        except FileNotFoundError:
            print(f"[ERROR] Dataset file not found: {file_path}")
            self.rows = []

        if self.rows and isinstance(self.rows[0], dict):
            self.header_fields = list(self.rows[0].keys())

        if multiplier > 1:
            self.rows = self.rows * int(multiplier)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        row = self.rows[idx]

        if self.input_mode == "nmo":
            return self._to_structured_text(row, mode="nmo")
        if self.input_mode == "flat_field":
            return self._to_structured_text(row, mode="flat_field")

        # For raw syslog-style lines, strip a leading tag prefix when present.
        line = row if isinstance(row, str) else ",".join(str(row.get(k, "") or "") for k in self.header_fields)
        if "<" in line and ">" in line and ":" in line:
            line = line.split(":", 1)[-1].strip()
        return line

    @staticmethod
    def _looks_like_header(line: str) -> bool:
        lower = line.lower()
        keys = ["name", "type", "path", "desc", "description", "examples", "example", "ex"]
        return any(k in lower for k in keys)

    @classmethod
    def _normalize_header(cls, value: str) -> str:
        return str(value or "").replace("\ufeff", "").strip().lower()

    @classmethod
    def _load_rows(cls, f) -> List[Any]:
        lines = [line.rstrip("\n") for line in f if line.strip()]
        if not lines:
            return []
        if cls._looks_like_header(lines[0]):
            reader = csv.DictReader(lines)
            return [dict(row) for row in reader if row]
        return lines

    @staticmethod
    def _csv_split(line: str) -> List[str]:
        try:
            return next(csv.reader([line]))
        except (csv.Error, StopIteration):
            return [x.strip() for x in line.split(",")]

    @classmethod
    def _row_to_field_dict(cls, row: Dict[str, Any]) -> Dict[str, str]:
        normalized = {cls._normalize_header(k): str(v or "") for k, v in row.items()}
        return {
            "name": normalized.get("name", ""),
            "desc": normalized.get("desc", "") or normalized.get("description", ""),
            "path": normalized.get("path", ""),
            "type": normalized.get("type", ""),
            "example": normalized.get("example", "") or normalized.get("examples", "") or normalized.get("ex", ""),
        }

    def _to_structured_text(self, row: Any, mode: str = "nmo") -> str:
        # Map CSV columns into a canonical field dict, then serialize it.
        if isinstance(row, dict):
            field_dict = self._row_to_field_dict(row)
        else:
            line = row
            parts = self._csv_split(line)
            if len(parts) >= 5:
                field_dict = {
                    "name": parts[0],
                    "desc": parts[1],
                    "path": parts[2],
                    "type": parts[3],
                    "example": parts[4],
                }
            elif len(parts) == 4:
                field_dict = {
                    "name": parts[0],
                    "desc": parts[1],
                    "path": parts[2],
                    "type": parts[3],
                    "example": "",
                }
            elif len(parts) == 3:
                field_dict = {
                    "name": parts[0],
                    "desc": "",
                    "path": parts[2],
                    "type": parts[1],
                    "example": "",
                }
            else:
                return line

        if not any(field_dict.values()):
            return ""

        return to_nmo_string(
            field_dict,
            mask_name=self.mask_name,
            input_mode=mode,
            drop_type=self.drop_type,
            drop_path=self.drop_path,
            drop_desc=self.drop_desc,
            drop_example=self.drop_example,
            no_placeholder=self.no_placeholder,
        )


def train_one_batch(
    encoder: nn.Module,
    teacher_encoder: nn.Module | None,
    optimizer: AdamW,
    scheduler: Any,
    batch: List[str],
    device: str,
    ablation: str,
    use_augment: bool,
    rtd_weight: float,
    varclr_weight: float,
    temperature: float,
    varclr_gamma: float,
    varclr_cov_weight: float,
    teacher_anchor_weight: float,
    grad_clip: float,
) -> Tuple[float, float, float, float, float]:
    # One update step combines contrastive loss with optional RTD and VarCLR terms.
    encoder.train()
    texts = list(batch)
    texts_contrastive = build_contrastive_views(texts) if use_augment else texts
    varclr_progress = min(1.0, max(0.0, float(varclr_weight) / 0.05)) if varclr_weight > 0.0 else 0.0
    varclr_view_prob = 0.22 + (0.24 * varclr_progress)
    allow_short_varclr = varclr_progress >= 0.35
    texts_varclr = (
        build_varclr_views(
            texts,
            perturb_prob=varclr_view_prob,
            allow_short_names=allow_short_varclr,
        )
        if use_augment
        else texts
    )

    token_labels = None
    texts_rtd = texts
    if ablation in {"MD", "MDV"} and use_augment and hasattr(encoder, "mlm_augment"):
        texts_rtd, token_labels = encoder.mlm_augment(
            texts,
            device=device,
            mask_prob=0.15,
        )

    if ablation == "MDV" and hasattr(encoder, "encode_pair"):
        z1_var, z1 = encoder.encode_pair(texts, device=device)
        z2_var, z2 = encoder.encode_pair(texts_contrastive, device=device)
    else:
        z1 = encoder(texts, device=device)
        z2 = encoder(texts_contrastive, device=device)
        z1_var = z1
        z2_var = z2

    loss_con = nt_xent(z1, z2, temperature=temperature)
    total_loss = loss_con

    loss_rtd = torch.tensor(0.0, device=device)
    if ablation in {"MD", "MDV"} and use_augment and token_labels and hasattr(encoder, "rtd_loss"):
        try:
            loss_rtd = encoder.rtd_loss(texts, texts_rtd, token_labels, device=device)
            total_loss = total_loss + rtd_weight * loss_rtd
        except Exception as e:
            print(f"[WARN] RTD loss calculation failed: {e}")

    loss_var = torch.tensor(0.0, device=device)
    if ablation == "MDV":
        if hasattr(encoder, "encode_varclr"):
            z1_var = encoder.encode_varclr(texts, device=device)
            z2_var = encoder.encode_varclr(texts_varclr, device=device)
        elif hasattr(encoder, "encode_pair"):
            z1_var, _ = encoder.encode_pair(texts, device=device)
            z2_var, _ = encoder.encode_pair(texts_varclr, device=device)
        loss_var = varclr_regularizer(
            z1_var,
            z2_var,
            gamma=varclr_gamma,
            cov_weight=varclr_cov_weight,
        )
        total_loss = total_loss + varclr_weight * loss_var

    loss_anchor = torch.tensor(0.0, device=device)
    if (
        ablation == "MDV"
        and teacher_encoder is not None
        and teacher_anchor_weight > 0.0
    ):
        with torch.no_grad():
            teacher_z = teacher_encoder(texts, device=device)
        student_z = F.normalize(z1, p=2, dim=1)
        teacher_z = F.normalize(teacher_z, p=2, dim=1)
        anchor_per_sample = 1.0 - torch.sum(student_z * teacher_z, dim=1)
        anchor_scale = 1.0 - (0.35 * varclr_progress)
        if use_augment and texts_varclr:
            changed_mask = torch.tensor(
                [1.0 if a != b else 0.15 for a, b in zip(texts, texts_varclr)],
                device=device,
                dtype=anchor_per_sample.dtype,
            )
            loss_anchor = (anchor_per_sample * changed_mask).mean() * anchor_scale
        else:
            loss_anchor = anchor_per_sample.mean() * anchor_scale
        total_loss = total_loss + teacher_anchor_weight * loss_anchor

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return (
        float(loss_con.item()),
        float(loss_rtd.item()),
        float(loss_var.item()),
        float(loss_anchor.item()),
        float(total_loss.item()),
    )


def get_varclr_weight(
    base_weight: float,
    epoch: int,
    total_epochs: int,
    start_epoch: int | None = 1,
    warmup_epochs: int = 0,
) -> float:
    if base_weight <= 0.0:
        return 0.0
    if start_epoch is None:
        start_epoch = 1
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_weight)
    progress_epoch = epoch - start_epoch + 1
    if progress_epoch >= warmup_epochs:
        return float(base_weight)
    return float(base_weight) * (float(progress_epoch) / float(warmup_epochs))


AUGMENT_ALIAS_GROUPS = [
    ["source", "src", "client", "origin"],
    ["destination", "dst", "dest", "server", "target"],
    ["user", "usr", "account"],
    ["session", "sess"],
    ["protocol", "proto"],
    ["message", "msg"],
    ["service", "svc"],
    ["address", "addr"],
]

AUGMENT_ALIAS_MAP = {
    token: [alt for alt in group if alt != token]
    for group in AUGMENT_ALIAS_GROUPS
    for token in group
}

PROTECTED_NAME_TOKENS = {
    "ip", "id", "bps", "pps", "rx", "tx", "tcp", "udp", "icmp",
    "nat", "dnat", "snat", "vpn", "ssl", "dns", "fw", "cpu",
    "mem", "rtt", "pkt", "pkts",
}


def _extract_nmo_component(text: str, component: str) -> str:
    pattern = rf"\[{component}\]\s*(.*?)(?=\s*\[[A-Z]+\]|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _replace_nmo_component(text: str, component: str, value: str) -> str:
    pattern = rf"(\[{component}\]\s*)(.*?)(?=\s*\[[A-Z]+\]|$)"
    return re.sub(pattern, lambda m: f"{m.group(1)}{value.strip()}", text, count=1)


def _augment_component_tokens(tokens: List[str], component: str) -> List[str]:
    if not tokens:
        return tokens
    out = list(tokens)

    if component == "TYPE":
        return out

    if component == "PATH":
        # all_log3 is path-heavy; perturbing PATH hurts more than it helps.
        return out

    for idx, tok in enumerate(list(out)):
        low = tok.lower()
        if low in PROTECTED_NAME_TOKENS:
            continue
        if low in AUGMENT_ALIAS_MAP and random.random() < 0.20:
            out[idx] = random.choice(AUGMENT_ALIAS_MAP[low])

    if component == "NAME" and len(out) > 2 and random.random() < 0.15:
        drop_idx = random.randrange(len(out))
        out.pop(drop_idx)

    if component == "NAME" and len(out) == 1 and len(out[0]) > 8 and random.random() < 0.10:
        token = out[0]
        out[0] = token[: max(4, int(len(token) * 0.90))]

    return out or tokens[:1]


def _augment_component_tokens_varclr(
    tokens: List[str],
    component: str,
    perturb_prob: float = 0.35,
    allow_short_names: bool = False,
) -> List[str]:
    if not tokens:
        return tokens
    out = list(tokens)

    if component in {"TYPE", "PATH"}:
        return out

    # Keep the VarCLR view close to the anchor text, but still allow enough
    # semantic movement to improve top-k recall beyond the exact top-1 match.
    if len(out) <= 1:
        return out
    if len(out) == 2 and not allow_short_names:
        return out

    candidate_indices: List[int] = []
    for idx, tok in enumerate(out):
        low = tok.lower()
        if low in PROTECTED_NAME_TOKENS:
            continue
        if low in AUGMENT_ALIAS_MAP:
            candidate_indices.append(idx)

    # Skip only when there is no safe alias candidate at all.
    if len(candidate_indices) < 1:
        return out

    # Perturb a moderate minority of samples and still change at most one
    # token so the view remains softer than the contrastive branch.
    if random.random() >= perturb_prob:
        return out

    idx = random.choice(candidate_indices)
    out[idx] = random.choice(AUGMENT_ALIAS_MAP[out[idx].lower()])

    # For very safe, longer names, occasionally perturb one more alias token
    # so VarCLR can still expand top-k neighborhoods in later epochs.
    if allow_short_names and len(out) >= 4 and len(candidate_indices) >= 2 and random.random() < 0.15:
        alt_indices = [i for i in candidate_indices if i != idx]
        if alt_indices:
            idx2 = random.choice(alt_indices)
            out[idx2] = random.choice(AUGMENT_ALIAS_MAP[out[idx2].lower()])

    # VarCLR view should be softer than the contrastive view: no token drop,
    # no truncation, just light alias perturbation on NAME.
    return out


def _varclr_sample_is_safe(text: str) -> bool:
    s = str(text or "")
    if "[NAME]" not in s or "[PATH]" not in s:
        return False
    name = _extract_nmo_component(s, "NAME")
    path = _extract_nmo_component(s, "PATH")
    name_tokens = [p for p in re.split(r"[\s_]+", name) if p]
    path_tokens = [p for p in re.split(r"[\s_\.]+", path) if p]
    if len(name_tokens) <= 1:
        return False
    protected = sum(1 for tok in name_tokens if tok.lower() in PROTECTED_NAME_TOKENS)
    if protected >= len(name_tokens):
        return False
    # all_log3 is path-heavy; only perturb samples where NAME still carries
    # enough signal relative to PATH.
    if len(path_tokens) <= 3:
        return len(name_tokens) >= 2
    return len(name_tokens) >= min(3, max(2, len(path_tokens) // 2))


def build_contrastive_views(texts: List[str]) -> List[str]:
    augmented: List[str] = []
    for text in texts:
        s = str(text or "")
        if "[NAME]" in s and "[TYPE]" in s and "[PATH]" in s:
            current = _extract_nmo_component(s, "NAME")
            if current:
                parts = [p for p in re.split(r"[\s_]+", current) if p]
                augmented_parts = _augment_component_tokens(parts, "NAME")
                if augmented_parts:
                    s = _replace_nmo_component(s, "NAME", " ".join(augmented_parts))
        augmented.append(s)
    return augmented


def build_varclr_views(
    texts: List[str],
    perturb_prob: float = 0.35,
    allow_short_names: bool = False,
) -> List[str]:
    augmented: List[str] = []
    for text in texts:
        s = str(text or "")
        if not _varclr_sample_is_safe(s):
            augmented.append(s)
            continue
        if "[NAME]" in s and "[TYPE]" in s and "[PATH]" in s:
            current = _extract_nmo_component(s, "NAME")
            if current:
                parts = [p for p in re.split(r"[\s_]+", current) if p]
                augmented_parts = _augment_component_tokens_varclr(
                    parts,
                    "NAME",
                    perturb_prob=perturb_prob,
                    allow_short_names=allow_short_names,
                )
                if augmented_parts:
                    s = _replace_nmo_component(s, "NAME", " ".join(augmented_parts))
        augmented.append(s)
    return augmented


def main():
    parser = argparse.ArgumentParser(
        description="Train M, MD, or MDV models for semantic mapping.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--csv_files", nargs="+", required=True, help="Input CSV/text files.")
    parser.add_argument("--outdir", required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--ckptM", type=str, default=None, help="Path to pre-trained M model for MD init.")
    parser.add_argument("--ckptMD", type=str, default=None, help="Path to pre-trained MD model for MDV init.")
    parser.add_argument("--encoding", type=str, default="utf-8", help="Input file encoding.")

    parser.add_argument("--input_mode", choices=["raw_msg", "flat_field", "nmo", "msg"], default="nmo")
    parser.add_argument("--ablation", choices=["M", "MD", "MDV"], required=True)
    parser.add_argument("--encoder_model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--mlm_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--use_augment", action="store_true")

    parser.add_argument("--mask_name", action="store_true")
    parser.add_argument("--drop_type", action="store_true")
    parser.add_argument("--drop_path", action="store_true")
    parser.add_argument("--drop_desc", action="store_true")
    parser.add_argument("--drop_example", action="store_true")
    parser.add_argument("--no_placeholder", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--rtd_weight", type=float, default=0.5)
    parser.add_argument("--varclr_weight", type=float, default=0.01)
    parser.add_argument("--varclr_gamma", type=float, default=1.0)
    parser.add_argument("--varclr_cov_weight", type=float, default=0.25)
    parser.add_argument("--teacher_anchor_weight", type=float, default=0.20)
    parser.add_argument("--varclr_start_epoch", type=int, default=3)
    parser.add_argument("--varclr_warmup_epochs", type=int, default=4)
    parser.add_argument("--dedup_texts", type=str2bool, default=True)
    parser.add_argument("--use_varclr_projector", type=str2bool, default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--data_multiplier", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)

    args = parser.parse_args()

    if args.input_mode == "msg":
        args.input_mode = "raw_msg"

    os.makedirs(args.outdir, exist_ok=True)

    run_config_path = os.path.join(args.outdir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    print(f"[INFO] Setting up '{args.ablation}' model...")
    model: nn.Module
    teacher_encoder: nn.Module | None = None

    if args.ablation == "M":
        model = MPNetEncoder(model_name=args.encoder_model, max_length=args.max_len)
    elif args.ablation == "MD":
        # MD starts from the base encoder and adds DiffCSE/RTD components.
        model = DiffCSEEncoder(
            encoder_model_name=args.encoder_model,
            mlm_model_name=args.mlm_model,
            max_length=args.max_len,
        )
        if args.ckptM and os.path.exists(args.ckptM):
            print(f"[INFO] Loading weights from M checkpoint: {args.ckptM}")
            ckpt = torch.load(args.ckptM, map_location="cpu")
            safe_load_state_dict(model.encoder, _extract_state_dict(ckpt))
    else:
        # MDV extends MD with the additional VarCLR branch.
        model = DiffCLREncoder(
            encoder_model_name=args.encoder_model,
            mlm_model_name=args.mlm_model,
            max_length=args.max_len,
            use_varclr_projector=args.use_varclr_projector,
        )
        if args.ckptMD and os.path.exists(args.ckptMD):
            print(f"[INFO] Loading weights from MD checkpoint: {args.ckptMD}")
            ckpt = torch.load(args.ckptMD, map_location="cpu")
            safe_load_state_dict(model, _extract_state_dict(ckpt))
        if args.teacher_anchor_weight > 0.0:
            teacher_encoder = copy.deepcopy(model.encoder)
            teacher_encoder.to(device)
            teacher_encoder.eval()
            for param in teacher_encoder.parameters():
                param.requires_grad = False

    model.to(device)

    print("[INFO] Loading data...")
    texts: List[str] = []
    for csv_file in args.csv_files:
        dataset = CsvLogDataset(
            csv_file,
            input_mode=args.input_mode,
            encoding=args.encoding,
            multiplier=args.data_multiplier,
            mask_name=args.mask_name,
            drop_type=args.drop_type,
            drop_path=args.drop_path,
            drop_desc=args.drop_desc,
            drop_example=args.drop_example,
            no_placeholder=args.no_placeholder,
        )
        texts.extend(dataset[i] for i in range(len(dataset)))

    if not texts:
        print("[ERROR] No training data found. Aborting.")
        return

    if args.dedup_texts:
        before = len(texts)
        texts = [t for t in dict.fromkeys(str(t).strip() for t in texts) if t]
        print(f"[INFO] Deduplicated training samples: {before} -> {len(texts)}")

    print(f"[INFO] Total training samples: {len(texts)}")

    # Drop the last batch only when there are enough samples to form a full batch.
    drop_last = len(texts) >= args.batch
    data_loader = DataLoader(
        texts,
        batch_size=args.batch,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("[ERROR] No trainable parameters found. Aborting.")
        return

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Trainable parameters: {n_trainable:,}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, len(data_loader) * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(
        f"[INFO] Starting training: mode={args.ablation}, "
        f"epochs={args.epochs}, steps={total_steps}, input_mode={args.input_mode}"
    )

    step = 0
    metrics_rows: List[Dict[str, float]] = []

    for ep in range(1, args.epochs + 1):
        total_loss_con = 0.0
        total_loss_rtd = 0.0
        total_loss_var = 0.0
        total_loss_anchor = 0.0
        total_loss_sum = 0.0
        current_varclr_weight = args.varclr_weight
        if args.ablation == "MDV":
            current_varclr_weight = get_varclr_weight(
                args.varclr_weight,
                ep,
                total_epochs=args.epochs,
                start_epoch=args.varclr_start_epoch,
                warmup_epochs=args.varclr_warmup_epochs,
            )

        for batch_data in data_loader:
            losses = train_one_batch(
                encoder=model,
                teacher_encoder=teacher_encoder,
                optimizer=optimizer,
                scheduler=scheduler,
                batch=batch_data,
                device=str(device),
                ablation=args.ablation,
                use_augment=args.use_augment,
                rtd_weight=args.rtd_weight,
                varclr_weight=current_varclr_weight,
                temperature=args.temperature,
                varclr_gamma=args.varclr_gamma,
                varclr_cov_weight=args.varclr_cov_weight,
                teacher_anchor_weight=args.teacher_anchor_weight,
                grad_clip=args.grad_clip,
            )

            total_loss_con += losses[0]
            total_loss_rtd += losses[1]
            total_loss_var += losses[2]
            total_loss_anchor += losses[3]
            total_loss_sum += losses[4]
            step += 1

            if args.save_every and args.save_every > 0 and (step % args.save_every == 0):
                # Optional step-level checkpointing for long runs.
                save_path = os.path.join(args.outdir, f"encoder_step_{step}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Saved checkpoint to {save_path}")

        num_batches = max(1, len(data_loader))
        avg_loss = total_loss_sum / num_batches
        avg_con = total_loss_con / num_batches
        avg_rtd = total_loss_rtd / num_batches
        avg_var = total_loss_var / num_batches
        avg_anchor = total_loss_anchor / num_batches

        row = {
            "epoch": ep,
            "avg_total_loss": avg_loss,
            "avg_contrastive_loss": avg_con,
            "avg_rtd_loss": avg_rtd,
            "avg_varclr_loss": avg_var,
            "avg_anchor_loss": avg_anchor,
        }
        metrics_rows.append(row)

        log_msg = f"[Epoch {ep}/{args.epochs}] Avg Loss: {avg_loss:.4f} (Con: {avg_con:.4f}"
        if args.ablation in {"MD", "MDV"}:
            log_msg += f", RTD: {avg_rtd:.4f}"
        if args.ablation == "MDV":
            log_msg += f", VarCLR: {avg_var:.4f}, VarW: {current_varclr_weight:.4f}"
            if args.teacher_anchor_weight > 0.0:
                log_msg += f", Anchor: {avg_anchor:.4f}"
        log_msg += ")"
        print(log_msg)

        # Save one checkpoint per epoch so later stages can reuse intermediate models.
        epoch_ckpt_path = os.path.join(args.outdir, f"encoder_epoch_{ep}.pt")
        torch.save(model.state_dict(), epoch_ckpt_path)

    metrics_path = os.path.join(args.outdir, "train_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "avg_total_loss",
                "avg_contrastive_loss",
                "avg_rtd_loss",
                "avg_varclr_loss",
                "avg_anchor_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    final_save_path = os.path.join(args.outdir, "encoder_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"[INFO] Training complete. Final checkpoint saved to {final_save_path}")
    print(f"[INFO] Run config saved to {run_config_path}")
    print(f"[INFO] Train metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
