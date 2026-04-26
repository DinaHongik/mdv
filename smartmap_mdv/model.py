# smartmap_mdv/model.py
from __future__ import annotations
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


class MPNetEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-9)
        return summed / denom

    def forward(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        out = self.model(**batch)
        emb = self._mean_pooling(out.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def encode(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        return self.forward(texts, device=device)


class TokenLevelDiscriminator(nn.Module):
    """Conditional RTD: detects replaced tokens using sentence embedding"""
    def __init__(self, hidden_size: int = 768, condition_size: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden_size + condition_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor,
                sentence_embedding: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, H = token_embeddings.shape
        sent_expanded = sentence_embedding.unsqueeze(1).expand(B, L, -1)
        combined = torch.cat([token_embeddings, sent_expanded], dim=-1)
        hidden = torch.tanh(self.proj(combined))
        logits = self.classifier(hidden).squeeze(-1)
        return logits


class DiffCSEEncoder(nn.Module):
    """DiffCSE: frozen generator + trainable conditional discriminator"""
    def __init__(self, mlm_model_name: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.encoder = MPNetEncoder()

        # Generator: FROZEN
        self.generator_tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
        self.generator = AutoModelForMaskedLM.from_pretrained(mlm_model_name)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()  

        gen_hidden_size = int(self.generator.config.hidden_size)
        enc_hidden_size = int(getattr(self.encoder.model.config, "hidden_size", 768))

        self.discriminator = TokenLevelDiscriminator(
            hidden_size=gen_hidden_size,
            condition_size=enc_hidden_size
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.generator.eval()
        return self

    def forward(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        return self.encoder(texts, device=device)

    def encode(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        return self.forward(texts, device=device)

    @torch.no_grad()
    def mlm_augment(self, texts: List[str], device: Optional[str] = None,
                    mask_prob: float = 0.15, top_k: int = 50) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate augmented view using FROZEN generator"""
        if device is None:
            device = next(self.parameters()).device

        augmented = []
        all_labels = []

        for text in texts:
            try:
                inputs = self.generator_tokenizer(
                    text, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)

                input_ids = inputs["input_ids"][0]
                labels = torch.ones_like(input_ids)

                special = {
                    self.generator_tokenizer.cls_token_id,
                    self.generator_tokenizer.sep_token_id,
                    self.generator_tokenizer.pad_token_id,
                    self.generator_tokenizer.bos_token_id if self.generator_tokenizer.bos_token_id else -1,
                    self.generator_tokenizer.eos_token_id if self.generator_tokenizer.eos_token_id else -1,
                }

                pos = [i for i, tok in enumerate(input_ids) if int(tok) not in special]

                if not pos:
                    augmented.append(text)
                    all_labels.append(labels)
                    continue

                n_mask = max(1, int(len(pos) * mask_prob))
                chosen = random.sample(pos, min(n_mask, len(pos)))

                masked = input_ids.clone()
                for p in chosen:
                    masked[p] = self.generator_tokenizer.mask_token_id

                out = self.generator(
                    input_ids=masked.unsqueeze(0),
                    attention_mask=inputs["attention_mask"]
                )
                logits = out.logits[0]  # [L, V]

                for p in chosen:
                    topk = torch.topk(logits[p], k=min(int(top_k), logits.size(-1)))
                    probs = F.softmax(topk.values, dim=-1)

                    idx = int(torch.multinomial(probs, 1).item())
                    new_token = int(topk.indices[idx].item())

                    if new_token != int(input_ids[p].item()):
                        labels[p] = 0
                    masked[p] = new_token

                aug_text = self.generator_tokenizer.decode(masked, skip_special_tokens=True)
                augmented.append(aug_text)
                all_labels.append(labels)

            except Exception:
                augmented.append(text)
                dummy_labels = torch.ones(512, device=device)
                all_labels.append(dummy_labels)

        return augmented, all_labels

    def rtd_loss(self, original_texts: List[str], augmented_texts: List[str],
                 token_labels: List[torch.Tensor], device: Optional[str] = None) -> torch.Tensor:
        """Conditional RTD loss: encoder + discriminator are trained, generator is frozen"""
        if device is None:
            device = next(self.parameters()).device

        sent_emb = self.encoder(augmented_texts, device=device)

        inputs = self.generator_tokenizer(
            augmented_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            base_model = None
            for attr in ['bert', 'roberta', 'electra', 'deberta', 'model']:
                if hasattr(self.generator, attr):
                    base_model = getattr(self.generator, attr)
                    break

            if base_model is None:
                gen_out = self.generator(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
                token_emb = gen_out.hidden_states[-1]
            else:
                gen_out = base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=False,
                    return_dict=True
                )
                token_emb = gen_out.last_hidden_state

        logits = self.discriminator(token_emb, sent_emb, inputs["attention_mask"])  # [B, L]

        mask = inputs["attention_mask"].float()
        B, L = logits.shape
        padded_labels = torch.ones(B, L, device=device)
        for i, label in enumerate(token_labels):
            length = min(label.numel(), L)
            padded_labels[i, :length] = label[:length].to(device)

        loss = F.binary_cross_entropy_with_logits(logits, padded_labels, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp_min(1e-9)
        return loss


class DiffCLREncoder(DiffCSEEncoder):
    """DiffCLR: DiffCSE + VarCLR"""
    def __init__(self, mlm_model_name: str = "bert-base-multilingual-cased"):
        super().__init__(mlm_model_name=mlm_model_name)
        self.mode = "MDV"
        
    def forward_varclr(self, texts: List[str], device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
            
        z1 = self.encoder(texts, device=device)
        
        texts_aug, _ = self.mlm_augment(texts, device=device, mask_prob=0.20)
        
        z2 = self.encoder(texts_aug, device=device)
        
        return z1, z2
