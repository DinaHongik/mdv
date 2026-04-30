from __future__ import annotations
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


class MPNetEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_length: int = 512,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-9)
        return summed / denom

    def encode_pair(
        self,
        texts: List[str],
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        out = self.model(**batch)
        emb_raw = self._mean_pooling(out.last_hidden_state, batch["attention_mask"])
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        return emb_raw, emb_norm

    def forward(
        self,
        texts: List[str],
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        emb_raw, emb_norm = self.encode_pair(texts, device=device)
        return emb_norm if normalize else emb_raw

    def encode(
        self,
        texts: List[str],
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.forward(texts, device=device, normalize=normalize)


class TokenLevelDiscriminator(nn.Module):
    """Conditional RTD: detects replaced tokens using sentence embedding."""
    def __init__(self, hidden_size: int = 768, condition_size: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden_size + condition_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        sentence_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, H = token_embeddings.shape
        sent_expanded = sentence_embedding.unsqueeze(1).expand(B, L, -1)
        combined = torch.cat([token_embeddings, sent_expanded], dim=-1)
        hidden = torch.tanh(self.proj(combined))
        logits = self.classifier(hidden).squeeze(-1)
        return logits


class DiffCSEEncoder(nn.Module):
    """DiffCSE: frozen generator + trainable conditional discriminator."""
    def __init__(
        self,
        encoder_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        mlm_model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length

        # Trainable sentence encoder
        self.encoder = MPNetEncoder(
            model_name=encoder_model_name,
            max_length=max_length,
        )

        # Frozen generator
        self.generator_tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
        self.generator = AutoModelForMaskedLM.from_pretrained(mlm_model_name)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

        gen_hidden_size = int(self.generator.config.hidden_size)
        enc_hidden_size = int(getattr(self.encoder.model.config, "hidden_size", 768))

        self.discriminator = TokenLevelDiscriminator(
            hidden_size=gen_hidden_size,
            condition_size=enc_hidden_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep generator frozen in eval mode
        self.generator.eval()
        return self

    def forward(
        self,
        texts: List[str],
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.encoder(texts, device=device, normalize=normalize)

    def encode(
        self,
        texts: List[str],
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.forward(texts, device=device, normalize=normalize)

    def encode_pair(
        self,
        texts: List[str],
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode_pair(texts, device=device)

    @torch.no_grad()
    def mlm_augment(
        self,
        texts: List[str],
        device: Optional[str] = None,
        mask_prob: float = 0.15,
        top_k: int = 50,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate augmented view using frozen MLM generator."""
        if device is None:
            device = next(self.parameters()).device

        augmented = []
        all_labels = []

        pad_id = self.generator_tokenizer.pad_token_id
        cls_id = self.generator_tokenizer.cls_token_id
        sep_id = self.generator_tokenizer.sep_token_id
        bos_id = self.generator_tokenizer.bos_token_id
        eos_id = self.generator_tokenizer.eos_token_id
        mask_id = self.generator_tokenizer.mask_token_id

        special = {
            cls_id if cls_id is not None else -1,
            sep_id if sep_id is not None else -1,
            pad_id if pad_id is not None else -1,
            bos_id if bos_id is not None else -1,
            eos_id if eos_id is not None else -1,
        }

        for text in texts:
            try:
                inputs = self.generator_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(device)

                input_ids = inputs["input_ids"][0]
                labels = torch.ones_like(input_ids, dtype=torch.float, device=device)

                pos = [i for i, tok in enumerate(input_ids) if int(tok) not in special]

                if not pos or mask_id is None:
                    augmented.append(text)
                    all_labels.append(labels)
                    continue

                n_mask = max(1, int(len(pos) * mask_prob))
                chosen = random.sample(pos, min(n_mask, len(pos)))

                masked = input_ids.clone()
                for p in chosen:
                    masked[p] = mask_id

                out = self.generator(
                    input_ids=masked.unsqueeze(0),
                    attention_mask=inputs["attention_mask"],
                )
                logits = out.logits[0]  # [L, V]

                for p in chosen:
                    topk = torch.topk(logits[p], k=min(int(top_k), logits.size(-1)))
                    probs = F.softmax(topk.values, dim=-1)

                    idx = int(torch.multinomial(probs, 1).item())
                    new_token = int(topk.indices[idx].item())

                    if new_token != int(input_ids[p].item()):
                        labels[p] = 0.0
                    masked[p] = new_token

                aug_text = self.generator_tokenizer.decode(masked, skip_special_tokens=True)
                augmented.append(aug_text)
                all_labels.append(labels)

            except Exception:
                augmented.append(text)
                dummy_labels = torch.ones(self.max_length, dtype=torch.float, device=device)
                all_labels.append(dummy_labels)

        return augmented, all_labels

    def rtd_loss(
        self,
        original_texts: List[str],
        augmented_texts: List[str],
        token_labels: List[torch.Tensor],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Conditional RTD loss: encoder + discriminator are trained, generator is frozen."""
        if device is None:
            device = next(self.parameters()).device

        sent_emb = self.encoder(augmented_texts, device=device)

        inputs = self.generator_tokenizer(
            augmented_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            base_model = None
            for attr in ["bert", "roberta", "electra", "deberta", "model"]:
                if hasattr(self.generator, attr):
                    base_model = getattr(self.generator, attr)
                    break

            if base_model is None:
                gen_out = self.generator(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                token_emb = gen_out.hidden_states[-1]
            else:
                gen_out = base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=False,
                    return_dict=True,
                )
                token_emb = gen_out.last_hidden_state

        logits = self.discriminator(token_emb, sent_emb, inputs["attention_mask"])  # [B, L]

        mask = inputs["attention_mask"].float()
        B, L = logits.shape
        padded_labels = torch.ones(B, L, dtype=torch.float, device=device)

        for i, label in enumerate(token_labels):
            length = min(label.numel(), L)
            padded_labels[i, :length] = label[:length].to(device)

        loss = F.binary_cross_entropy_with_logits(logits, padded_labels, reduction="none")
        loss = (loss * mask).sum() / mask.sum().clamp_min(1e-9)
        return loss


class DiffCLREncoder(DiffCSEEncoder):
    """DiffCLR: DiffCSE + VarCLR."""
    def __init__(
        self,
        encoder_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        mlm_model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
        use_varclr_projector: bool = False,
    ):
        super().__init__(
            encoder_model_name=encoder_model_name,
            mlm_model_name=mlm_model_name,
            max_length=max_length,
        )
        self.mode = "MDV"
        self.use_varclr_projector = bool(use_varclr_projector)
        enc_hidden_size = int(getattr(self.encoder.model.config, "hidden_size", 768))
        self.varclr_projector = None
        if self.use_varclr_projector:
            self.varclr_projector = nn.Sequential(
                nn.Linear(enc_hidden_size, enc_hidden_size),
                nn.GELU(),
                nn.LayerNorm(enc_hidden_size),
                nn.Linear(enc_hidden_size, enc_hidden_size),
            )

    def encode_varclr(
        self,
        texts: List[str],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        z_raw, _ = self.encoder.encode_pair(texts, device=device)
        if self.varclr_projector is None:
            return z_raw
        return self.varclr_projector(z_raw)

    def forward_varclr(
        self,
        texts: List[str],
        texts_aug: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device

        z1 = self.encode_varclr(texts, device=device)
        if texts_aug is None:
            texts_aug, _ = self.mlm_augment(texts, device=device, mask_prob=0.20)
        z2 = self.encode_varclr(texts_aug, device=device)

        return z1, z2
