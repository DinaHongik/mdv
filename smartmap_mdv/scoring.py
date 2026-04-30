from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from .model import DiffCLREncoder
from .data import NMOFields
from .baselines import BM25Encoder


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class IsotonicRegressionCalibrator:
    """Simple isotonic regression implementation for probability calibration."""
    
    def __init__(self):
        self.is_fitted = False
        self.x_values = None
        self.y_values = None
        
    def fit(self, scores: np.ndarray, correctness: np.ndarray):
        """Fit isotonic regression using PAVA algorithm."""
        # Combine scores and correctness
        pairs = list(zip(scores, correctness))
        pairs.sort(key=lambda x: x[0])
        
        x_sorted = np.array([p[0] for p in pairs])
        y_sorted = np.array([p[1] for p in pairs])
        
        # PAVA (Pool Adjacent Violators Algorithm)
        n = len(x_sorted)
        if n == 0:
            self.is_fitted = False
            return
            
        # Initialize with each point as its own block
        start = np.arange(n)
        end = np.arange(n) + 1
        weight = np.ones(n)
        sum_y = y_sorted.astype(float)
        
        # Merge violating blocks
        i = 0
        while i < len(start) - 1:
            current_avg = sum_y[i] / weight[i]
            next_avg = sum_y[i + 1] / weight[i + 1]
            
            if current_avg > next_avg:  # Violation found
                # Merge blocks
                sum_y[i + 1] += sum_y[i]
                weight[i + 1] += weight[i]
                start = np.delete(start, i)
                end = np.delete(end, i)
                sum_y = np.delete(sum_y, i)
                weight = np.delete(weight, i)
                i = max(0, i - 1)
            else:
                i += 1
        
        # Store fitted values
        self.x_values = x_sorted
        self.y_values = []
        
        for i in range(len(start)):
            avg_y = sum_y[i] / weight[i]
            self.y_values.extend([avg_y] * (end[i] - start[i]))
        
        self.y_values = np.array(self.y_values)
        self.is_fitted = True
        
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict calibrated values."""
        if not self.is_fitted or self.x_values is None or self.y_values is None or len(self.x_values) == 0:
            return scores

        calibrated = np.interp(scores, self.x_values, self.y_values)
        return calibrated


class IntegratedScorer:
    """
    Integrated scoring module for NMO-MDV framework.
    Computes S(i,j) = α·s_name(i,j) + β·s_type(i,j) + γ·s_path(i,j) + δ·s_lex(i,j)
    """
    
    def __init__(self, 
                 encoder: DiffCLREncoder,
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 1.0,
                 delta: float = 0.1,
                 device: str = "cuda"):
        self.encoder = encoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.device = device
        self.encoder.eval()

    def _extract_component(self, nmo_text: str, component: str) -> str:
        """Extract specific component from NMO text."""
        tag = f"[{component.upper()}]"
        if tag not in nmo_text:
            return ""
        try:
            return nmo_text.split(tag)[1].split("[")[0].strip()
        except IndexError:
            return ""

    def _parse_all_components(self, texts: List[str]) -> Dict[str, List[str]]:
        """Parse all NMO components from a list of texts in one pass."""
        components = {"name": [], "type": [], "path": [], "desc": []}
        for text in texts:
            components["name"].append(self._extract_component(text, "name"))
            components["type"].append(self._extract_component(text, "type"))
            components["path"].append(self._extract_component(text, "path"))
            components["desc"].append(self._extract_component(text, "desc"))
        return components

    def _compute_cosine_similarity(self, embeddings_a: torch.Tensor, 
                                  embeddings_b: torch.Tensor) -> np.ndarray:
        """Compute cosine similarity matrix between two embedding sets."""
        if embeddings_a.nelement() == 0 or embeddings_b.nelement() == 0:
            return np.zeros((embeddings_a.shape[0], embeddings_b.shape[0]))
        
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        return torch.mm(embeddings_a, embeddings_b.t()).cpu().numpy()
    
    def compute_component_similarities(self, 
                                     source_components: Dict[str, List[str]],
                                     target_components: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Compute cosine similarities for each NMO component efficiently.
        """
        num_sources = len(source_components["name"])
        num_targets = len(target_components["name"])
        similarities = {
            "name": np.zeros((num_sources, num_targets)),
            "type": np.zeros((num_sources, num_targets)),
            "path": np.zeros((num_sources, num_targets))
        }

        for component in ["name", "type", "path"]:
            source_comp_texts = source_components[component]
            target_comp_texts = target_components[component]
            
            valid_source_indices = [i for i, t in enumerate(source_comp_texts) if t]
            valid_target_indices = [i for i, t in enumerate(target_comp_texts) if t]
            
            if not valid_source_indices or not valid_target_indices:
                continue

            valid_source_texts = [source_comp_texts[i] for i in valid_source_indices]
            valid_target_texts = [target_comp_texts[i] for i in valid_target_indices]
            
            with torch.no_grad():
                source_emb = self.encoder.encode(valid_source_texts, device=self.device)
                target_emb = self.encoder.encode(valid_target_texts, device=self.device)
            
            sim_matrix = self._compute_cosine_similarity(source_emb, target_emb)
            
            ix = np.ix_(valid_source_indices, valid_target_indices)
            similarities[component][ix] = sim_matrix
            
        return similarities

    def compute_lexical_similarity(self, 
                                   source_components: Dict[str, List[str]],
                                   target_components: Dict[str, List[str]]) -> np.ndarray:
        """Computes BM25-based lexical similarity."""
        source_lex_texts = [f"{n} {d}" for n, d in zip(source_components["name"], source_components["desc"])]
        target_lex_texts = [f"{n} {d}" for n, d in zip(target_components["name"], target_components["desc"])]

        bm25 = BM25Encoder()
        bm25.fit(target_lex_texts)
        
        scores = np.zeros((len(source_lex_texts), len(target_lex_texts)))
        for i, query in enumerate(source_lex_texts):
            scores[i, :] = bm25.get_scores(query)
            
        # Normalize scores to [0, 1] for stable combination
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
            
        return scores

    def compute_integrated_scores(self,
                                 source_texts: List[str],
                                 target_texts: List[str]) -> np.ndarray:
        """
        Compute integrated scores S(i,j) for all source-target pairs.
        """
        source_components = self._parse_all_components(source_texts)
        target_components = self._parse_all_components(target_texts)

        # Compute semantic similarities
        semantic_similarities = self.compute_component_similarities(source_components, target_components)
        
        # Compute lexical similarity
        lexical_similarity = np.zeros_like(semantic_similarities["name"])
        if self.delta > 0:
            lexical_similarity = self.compute_lexical_similarity(source_components, target_components)

        # Combine with weights
        S = (self.alpha * semantic_similarities["name"] + 
             self.beta * semantic_similarities["type"] + 
             self.gamma * semantic_similarities["path"] +
             self.delta * lexical_similarity)
        
        return S
    
    def get_top_k_candidates(self,
                           source_texts: List[str],
                           target_texts: List[str],
                           k: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Get top-k target candidates for each source field.
        
        Args:
            source_texts: List of source NMO field texts
            target_texts: List of target NMO field texts
            k: Number of top candidates to return
            
        Returns:
            List of top-k candidates for each source field
        """
        S = self.compute_integrated_scores(source_texts, target_texts)
        
        top_k_results = []
        for i in range(len(source_texts)):
            scores = S[i]
            top_indices = np.argsort(scores)[::-1][:k]
            top_k_results.append([(int(idx), float(scores[idx])) for idx in top_indices])
        
        return top_k_results


class ProbabilityCalibrator:
    """
    Probability calibration module using Isotonic Regression.
    Calibrates integrated scores S(i,j) into well-calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegressionCalibrator()
        self.is_fitted = False
        
    def fit(self, scores: np.ndarray, correctness: np.ndarray):
        """
        Fit calibration model using top-1 confidence and correctness.
        
        Args:
            scores: Integrated scores S(i,j) for calibration
            correctness: Binary correctness indicators (1 if correct, 0 otherwise)
        """
        # Extract top-1 scores and correctness for calibration
        top1_scores = np.max(scores, axis=1)
        top1_correctness = correctness
        
        # Fit isotonic regression
        self.calibrator.fit(top1_scores, top1_correctness)
        self.is_fitted = True
        
    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Calibrate scores into probabilities.
        
        Args:
            scores: Integrated score matrix S
            
        Returns:
            Calibrated probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
            
        # Convert scores to probabilities via softmax
        probs = _softmax(scores, axis=1)
        
        # Calibrate top-1 probabilities
        top1_scores = np.max(scores, axis=1)
        top1_probs = probs[np.arange(len(scores)), np.argmax(scores, axis=1)]
        
        # Apply isotonic regression calibration
        calibrated_top1_probs = self.calibrator.predict(top1_scores)
        
        # Redistribute remaining probability mass
        scaling_factors = calibrated_top1_probs / (top1_probs + 1e-8)
        scaling_factors = np.clip(scaling_factors, 0.1, 10.0)  # Prevent extreme scaling
        
        calibrated_probs = probs.copy()
        for i in range(len(scores)):
            top1_idx = np.argmax(scores[i])
            calibrated_probs[i, top1_idx] = calibrated_top1_probs[i]
            
            # Redistribute remaining mass
            remaining_mass = 1.0 - calibrated_top1_probs[i]
            current_remaining = 1.0 - calibrated_probs[i, top1_idx]
            if current_remaining > 1e-8:
                other_indices = [j for j in range(len(scores[i])) if j != top1_idx]
                calibrated_probs[i, other_indices] *= (remaining_mass / current_remaining)
        
        return calibrated_probs
    
    def compute_ece(self, probs: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probs: Calibrated probabilities
            correctness: Binary correctness indicators
            n_bins: Number of bins for ECE computation
            
        Returns:
            ECE value
        """
        # Get top-1 confidence and predictions
        confidence = np.max(probs, axis=1)
        predicted = np.argmax(probs, axis=1)
        
        # Compute accuracy
        accuracy = correctness
        
        # Compute ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                confidence_in_bin = confidence[in_bin].mean()
                ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)
        
        return ece


class NMOMDVEvaluator:
    """
    Complete evaluator for NMO-MDV framework with integrated scoring and calibration.
    """
    
    def __init__(self, 
                 encoder: DiffCLREncoder,
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 1.0,
                 device: str = "cuda"):
        """
        Initialize complete evaluator.
        
        Args:
            encoder: Trained NMO-MDV encoder
            alpha, beta, gamma: Weights for integrated scoring
            device: Device for computation
        """
        self.scorer = IntegratedScorer(encoder, alpha, beta, gamma, device)
        self.calibrator = ProbabilityCalibrator()
        
    def evaluate_with_calibration(self,
                                 source_texts: List[str],
                                 target_texts: List[str],
                                 true_mappings: Optional[List[int]] = None) -> Dict:
        """
        Complete evaluation with integrated scoring and probability calibration.
        
        Args:
            source_texts: List of source NMO field texts
            target_texts: List of target NMO field texts
            true_mappings: Ground truth target indices for each source (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        # Compute integrated scores
        S = self.scorer.compute_integrated_scores(source_texts, target_texts)
        
        results = {
            "integrated_scores": S,
            "component_similarities": self.scorer.compute_component_similarities(source_texts, target_texts)
        }
        
        # If ground truth available, perform calibration
        if true_mappings is not None:
            correctness = np.array([1 if pred == true else 0 for pred, true in 
                                  zip(np.argmax(S, axis=1), true_mappings)])
            
            # Fit calibrator
            self.calibrator.fit(S, correctness)
            
            # Calibrate probabilities
            calibrated_probs = self.calibrator.calibrate_scores(S)
            
            # Compute metrics
            results.update({
                "calibrated_probabilities": calibrated_probs,
                "ece_before": self.calibrator.compute_ece(_softmax(S, axis=1), correctness),
                "ece_after": self.calibrator.compute_ece(calibrated_probs, correctness),
                "correctness": correctness
            })
        
        return results