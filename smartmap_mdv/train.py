# ------------------------------------------------------------
# Smart Mapping (M / MD / MDV)
# ------------------------------------------------------------
"""
This script handles the self-supervised training of the M, MD, and MDV models
for semantic schema mapping. It takes one or more CSV files as input, where
each row represents a log field to be used for training.

Modes of Operation:
- msg: Treats each CSV line as a raw context sentence.
- nmo: Serializes each CSV line into the NMO format ([NAME]...[TYPE]...etc.)
         before feeding it to the model.

Supported Ablations (via --ablation flag):
- M  : Trains a standard MPNet sentence encoder with contrastive loss (SimCSE-style).
- MD : Adds a Replaced Token Detection (RTD) task (DiffCSE-style).
- MDV: Adds VarCLR regularization on top of MD for more robust embeddings.

Usage Examples:
# M (MPNet) with NMO-serialized input
python train.py \n    --csv_files nmo_dataset/all_log.csv \n    --ablation M --epochs 10 --batch 8 --lr 2e-5 \n    --input_mode nmo --outdir out/M_nmo

# MD (MPNet+DiffCSE) with augmentation
python train.py \n    --csv_files nmo_dataset/all_log.csv \n    --ablation MD --epochs 15 --batch 8 --lr 1e-5 \n    --input_mode nmo --use_augment --outdir out/MD_nmo

# MDV (MPNet+DiffCSE+VarCLR) with augmentation
python train.py \n    --csv_files nmo_dataset/all_log.csv \n    --ablation MDV --epochs 20 --batch 8 --lr 5e-6 \n    --varclr_weight 0.05 --use_augment --input_mode nmo --outdir out/MDV_nmo
"""
import argparse
import csv
import os
import random
from typing import List, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from smartmap_mdv.data import to_nmo_string
from smartmap_mdv.losses import nt_xent, varclr_regularizer
from smartmap_mdv.model import MPNetEncoder, DiffCSEEncoder, DiffCLREncoder


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Safely extracts the state_dict from a checkpoint object."""
    if ckpt_obj is None:
        return {}
    if isinstance(ckpt_obj, dict):
        # Handle cases where the state_dict is nested
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        return ckpt_obj  # Assume the dict itself is the state_dict
    if hasattr(ckpt_obj, "state_dict") and callable(ckpt_obj.state_dict):
        return ckpt_obj.state_dict()
    return {}


def safe_load_state_dict(
    target_model: nn.Module, source_state_dict: Dict[str, torch.Tensor], strict: bool = False
) -> bool:
    """
    Safely loads a state_dict into a model, filtering out layers with mismatched shapes.
    """
    if not source_state_dict:
        return False
    try:
        target_model.load_state_dict(source_state_dict, strict=strict)
        return True
    except (RuntimeError, ValueError):
        # Fallback for shape mismatches
        target_dict = target_model.state_dict()
        filtered_dict = {
            k: v for k, v in source_state_dict.items() if k in target_dict and target_dict[k].shape == v.shape
        }
        if filtered_dict:
            target_model.load_state_dict(filtered_dict, strict=False)
            print(f"[WARN] Loaded state_dict with filtered layers due to shape mismatches.")
            return True
        print("[WARN] Could not load state_dict, no matching layers found.")
        return False


class CsvLogDataset(Dataset):
    """
    Reads a CSV file for self-supervised training.
    Each line in the CSV is treated as a log entry and can be converted
    to NMO format.
    """
    def __init__(self, file_path: str, input_mode: str = "msg", encoding: str = "utf-8", multiplier: int = 1):
        self.input_mode = input_mode
        try:
            with open(file_path, "r", encoding=encoding) as f:
                self.lines = [line.rstrip("\n") for line in f if line.strip()]
        except FileNotFoundError:
            print(f"[ERROR] Dataset file not found: {file_path}")
            self.lines = []

        if self.lines and self._looks_like_header(self.lines[0]):
            self.lines = self.lines[1:]

        if multiplier > 1:
            self.lines = self.lines * int(multiplier)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        line = self.lines[idx]
        if self.input_mode == "nmo":
            return self._to_nmo(line)

        # Basic cleanup for msg mode
        if "<" in line and ">" in line and ":" in line:
            line = line.split(":", 1)[-1].strip()
        return line

    @staticmethod
    def _looks_like_header(line: str) -> bool:
        """Heuristically checks if a line is a CSV header."""
        lower = line.lower()
        keys = ["name", "type", "path", "desc", "description", "examples", "ex"]
        return any(k in lower for k in keys)

    @staticmethod
    def _csv_split(line: str) -> List[str]:
        """Splits a CSV line, handling potential errors."""
        try:
            return next(csv.reader([line]))
        except (csv.Error, StopIteration):
            return [x.strip() for x in line.split(",")]

    def _to_nmo(self, line: str) -> str:
        """Converts a CSV line to a dictionary and then to an NMO string."""
        parts = self._csv_split(line)
        field_dict = {}
        if len(parts) >= 5:
            field_dict = {
                "name": parts[0], "desc": parts[1], "path": parts[2],
                "type": parts[3], "example": parts[4]
            }
        elif len(parts) == 4:
            field_dict = {
                "name": parts[0], "desc": parts[1], "path": parts[2],
                "type": parts[3], "example": ""
            }
        else:
            return line  # Cannot parse, return as is

        return to_nmo_string(field_dict, input_mode="nmo")


def train_one_batch(
    encoder: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    batch: List[str],
    device: str,
    ablation: str,
    use_augment: bool,
    rtd_weight: float,
    varclr_weight: float,
) -> Tuple[float, float, float, float]:
    """
    Performs a single training step for a batch of texts.

    Args:
        encoder: The model to train (M, MD, or MDV).
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        batch: A list of input texts.
        device: The device to train on ('cuda' or 'cpu').
        ablation: The training mode ('M', 'MD', 'MDV').
        use_augment: Whether to use MLM-based augmentation.
        rtd_weight: The weight for the Replaced Token Detection loss.
        varclr_weight: The weight for the VarCLR regularization loss.

    Returns:
        A tuple containing the contrastive, RTD, VarCLR, and total losses.
    """
    encoder.train()
    texts = list(batch)

    # --- Augmentation (for MD and MDV) ---
    token_labels = None
    if ablation in {"MD", "MDV"} and use_augment and hasattr(encoder, "mlm_augment"):
        mask_prob = 0.15 if ablation == "MD" else 0.20
        texts_aug, token_labels = encoder.mlm_augment(texts, device=device, mask_prob=mask_prob)
    else:
        texts_aug = texts

    # --- Forward Pass ---
    z1 = encoder(texts, device=device)
    z2 = encoder(texts_aug, device=device)

    # --- Loss Calculation ---
    loss_con = nt_xent(z1, z2)
    total_loss = loss_con

    loss_rtd = torch.tensor(0.0, device=device)
    if ablation in {"MD", "MDV"} and use_augment and token_labels and hasattr(encoder, "rtd_loss"):
        try:
            loss_rtd = encoder.rtd_loss(texts, texts_aug, token_labels, device=device)
            total_loss += rtd_weight * loss_rtd
        except Exception as e:
            print(f"[WARN] RTD loss calculation failed: {e}")

    loss_var = torch.tensor(0.0, device=device)
    if ablation == "MDV" and hasattr(encoder, "forward_varclr"):
        # For VarCLR, z1 and z2 should come from the dedicated forward pass
        z1_v, z2_v = encoder.forward_varclr(texts, device=device)
        loss_var = varclr_regularizer(z1_v, z2_v)
        total_loss += varclr_weight * loss_var

    # --- Backpropagation ---
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return (
        loss_con.item(),
        loss_rtd.item(),
        loss_var.item(),
        total_loss.item(),
    )


def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(
        description="Train M, MD, or MDV models for semantic mapping.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--csv_files", nargs="+", required=True, help="Path to one or more input CSV files for training.")
    parser.add_argument("--outdir", required=True, help="Directory to save checkpoints.")
    parser.add_argument("--ckptM", type=str, default=None, help="Path to a pre-trained M model to initialize MD.")
    parser.add_argument("--ckptMD", type=str, default=None, help="Path to a pre-trained MD model to initialize MDV.")

    # --- Model & Training Arguments ---
    parser.add_argument("--input_mode", choices=["nmo", "msg"], default="nmo", help="Input format: 'nmo' for serialized objects, 'msg' for raw text.")
    parser.add_argument("--ablation", choices=["M", "MD", "MDV"], required=True, help="Select model version to train.")
    parser.add_argument("--mlm_model", type=str, default="bert-base-multilingual-cased", help="MLM model for augmentation in MD/MDV.")
    parser.add_argument("--use_augment", action="store_true", help="Enable MLM-based data augmentation (for MD/MDV).")

    # --- Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--varclr_weight", type=float, default=0.05, help="Weight for VarCLR loss (for MDV).")
    parser.add_argument("--rtd_weight", type=float, default=0.5, help="Weight for RTD loss (for MD/MDV).")

    # --- System Arguments ---
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_multiplier", type=int, default=1, help="Factor to multiply dataset size by for longer training.")
    parser.add_argument("--save_every", type=int, default=0, help="Save a checkpoint every N steps. Set to 0 to disable.")

    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Model Selection and Initialization ---
    print(f"[INFO] Setting up '{args.ablation}' model...")
    model: nn.Module
    if args.ablation == "M":
        model = MPNetEncoder()
    elif args.ablation == "MD":
        model = DiffCSEEncoder(mlm_model_name=args.mlm_model)
        if args.ckptM and os.path.exists(args.ckptM):
            print(f"[INFO] Loading weights from M checkpoint: {args.ckptM}")
            ckpt = torch.load(args.ckptM, map_location="cpu")
            safe_load_state_dict(model.encoder, _extract_state_dict(ckpt))
    else:  # MDV
        model = DiffCLREncoder(mlm_model_name=args.mlm_model)
        if args.ckptMD and os.path.exists(args.ckptMD):
            print(f"[INFO] Loading weights from MD checkpoint: {args.ckptMD}")
            ckpt = torch.load(args.ckptMD, map_location="cpu")
            safe_load_state_dict(model, _extract_state_dict(ckpt))

    model.to(device)

    # --- Data Loading ---
    print("[INFO] Loading data...")
    texts: List[str] = []
    for csv_file in args.csv_files:
        dataset = CsvLogDataset(csv_file, input_mode=args.input_mode, multiplier=args.data_multiplier)
        texts.extend(dataset[i] for i in range(len(dataset)))

    if not texts:
        print("[ERROR] No training data found. Aborting.")
        return

    print(f"[INFO] Total training samples: {len(texts)}")
    data_loader = DataLoader(texts, batch_size=args.batch, shuffle=True, drop_last=True)

    # --- Optimizer and Scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("[ERROR] No trainable parameters found. Aborting.")
        return
    
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(data_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    # --- Training Loop ---
    print(f"[INFO] Starting training: {args.ablation} mode, {args.epochs} epochs, {total_steps} total steps.")
    step = 0
    for ep in range(1, args.epochs + 1):
        total_loss_con, total_loss_rtd, total_loss_var, total_loss_sum = 0.0, 0.0, 0.0, 0.0
        for batch_data in data_loader:
            losses = train_one_batch(
                model, optimizer, scheduler, batch_data, str(device),
                ablation=args.ablation, use_augment=args.use_augment,
                rtd_weight=args.rtd_weight, varclr_weight=args.varclr_weight
            )
            total_loss_con += losses[0]
            total_loss_rtd += losses[1]
            total_loss_var += losses[2]
            total_loss_sum += losses[3]
            step += 1

            if args.save_every and args.save_every > 0 and (step % args.save_every == 0):
                save_path = os.path.join(args.outdir, f"encoder_step_{step}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Saved checkpoint to {save_path}")

        # --- Epoch End Logging ---
        num_batches = len(data_loader)
        if num_batches > 0:
            avg_loss = total_loss_sum / num_batches
            avg_con = total_loss_con / num_batches
            avg_rtd = total_loss_rtd / num_batches
            avg_var = total_loss_var / num_batches
            
            log_msg = f"[Epoch {ep}/{args.epochs}] Avg Loss: {avg_loss:.4f} (Con: {avg_con:.4f}"
            if args.ablation in {"MD", "MDV"}:
                log_msg += f", RTD: {avg_rtd:.4f}"
            if args.ablation == "MDV":
                log_msg += f", VarCLR: {avg_var:.4f}"
            log_msg += ")"
            print(log_msg)

    # --- Final Checkpoint ---
    final_save_path = os.path.join(args.outdir, f"encoder_step_{step}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"[INFO] Training complete. Final checkpoint saved to {final_save_path}")



if __name__ == "__main__":
    main()
