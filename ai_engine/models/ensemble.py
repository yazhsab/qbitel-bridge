"""
QBITEL Engine - Ensemble Models

This module implements ensemble methods for combining multiple AI models
to improve prediction accuracy and robustness.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import concurrent.futures

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base import BaseModel, ModelInput, ModelOutput, ModelState
from ..core.config import Config
from ..core.exceptions import ModelException, EnsembleException


class EnsembleMethod(str, Enum):
    """Ensemble combination methods."""

    VOTING = "voting"  # Majority voting
    AVERAGING = "averaging"  # Average predictions
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average
    STACKING = "stacking"  # Meta-learner stacking
    BOOSTING = "boosting"  # Gradient boosting style
    MAX_CONFIDENCE = "max_confidence"  # Select prediction with highest confidence


class VotingType(str, Enum):
    """Voting types for classification."""

    HARD = "hard"  # Use predicted class labels
    SOFT = "soft"  # Use predicted probabilities


@dataclass
class EnsembleMember:
    """Individual ensemble member information."""

    model: BaseModel
    weight: float = 1.0
    is_active: bool = True
    performance_score: Optional[float] = None
    last_updated: Optional[float] = None


@dataclass
class EnsembleResult:
    """Ensemble prediction result."""

    final_prediction: torch.Tensor
    individual_predictions: Dict[str, torch.Tensor]
    individual_confidences: Dict[str, torch.Tensor]
    member_weights: Dict[str, float]
    combination_method: str
    consensus_score: Optional[float] = None
    uncertainty_score: Optional[float] = None


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models.

    This class provides various ensemble methods including voting,
    averaging, weighted combining, and stacking approaches.
    """

    def __init__(
        self,
        config: Config,
        model_name: str = "ensemble_model",
        ensemble_method: EnsembleMethod = EnsembleMethod.VOTING,
        voting_type: VotingType = VotingType.SOFT,
    ):
        """Initialize ensemble model."""
        super().__init__(config, model_name)

        self.ensemble_method = ensemble_method
        self.voting_type = voting_type

        # Ensemble members
        self.members: Dict[str, EnsembleMember] = {}
        self.member_order: List[str] = []  # Maintain insertion order

        # Meta-learner for stacking
        self.meta_learner: Optional[nn.Module] = None

        # Weights for weighted averaging
        self.adaptive_weights = False
        self.weight_update_frequency = 100  # Update weights every N predictions
        self.prediction_count = 0

        # Performance tracking
        self.ensemble_metrics = {
            "total_predictions": 0,
            "consensus_predictions": 0,
            "high_uncertainty_predictions": 0,
            "member_agreement_score": 0.0,
        }

        # Async execution
        self.max_workers = (
            config.training.max_workers
            if hasattr(config.training, "max_workers")
            else 4
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        self.logger.info(f"EnsembleModel initialized with method: {ensemble_method}")

    def add_member(
        self,
        member_model: BaseModel,
        member_name: str,
        weight: float = 1.0,
        performance_score: Optional[float] = None,
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            member_model: Model to add
            member_name: Unique name for the member
            weight: Initial weight for the member
            performance_score: Performance score for weight calculation
        """
        if member_name in self.members:
            raise EnsembleException(
                f"Member '{member_name}' already exists in ensemble"
            )

        member = EnsembleMember(
            model=member_model,
            weight=weight,
            performance_score=performance_score,
            last_updated=time.time(),
        )

        self.members[member_name] = member
        self.member_order.append(member_name)

        self.logger.info(f"Added ensemble member: {member_name} (weight: {weight})")

    def remove_member(self, member_name: str) -> bool:
        """Remove a model from the ensemble."""
        if member_name not in self.members:
            return False

        del self.members[member_name]
        self.member_order.remove(member_name)

        self.logger.info(f"Removed ensemble member: {member_name}")
        return True

    def set_member_weight(self, member_name: str, weight: float) -> bool:
        """Set weight for a specific member."""
        if member_name not in self.members:
            return False

        self.members[member_name].weight = weight
        self.members[member_name].last_updated = time.time()

        self.logger.info(f"Updated weight for {member_name}: {weight}")
        return True

    def set_member_active(self, member_name: str, is_active: bool) -> bool:
        """Activate or deactivate a member."""
        if member_name not in self.members:
            return False

        self.members[member_name].is_active = is_active
        self.logger.info(f"Set {member_name} active: {is_active}")
        return True

    def get_active_members(self) -> Dict[str, EnsembleMember]:
        """Get all active ensemble members."""
        return {
            name: member for name, member in self.members.items() if member.is_active
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble (not typically used directly)."""
        # This is mainly for interface compliance
        # Actual ensemble logic is in predict() method
        return x

    def predict(self, input_data: ModelInput) -> ModelOutput:
        """Make ensemble prediction."""
        start_time = time.time()

        try:
            # Validate input
            if not self.validate_input(input_data):
                raise EnsembleException("Input validation failed")

            # Get active members
            active_members = self.get_active_members()
            if not active_members:
                raise EnsembleException("No active ensemble members available")

            # Get predictions from all active members
            individual_results = self._get_individual_predictions(
                active_members, input_data
            )

            # Combine predictions based on ensemble method
            ensemble_result = self._combine_predictions(
                individual_results, active_members
            )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._update_ensemble_metrics(ensemble_result)
            self.update_inference_metrics(processing_time_ms, success=True)

            # Create output
            output = ModelOutput(
                predictions=ensemble_result.final_prediction,
                confidence=self._calculate_ensemble_confidence(ensemble_result),
                metadata={
                    "ensemble_method": self.ensemble_method.value,
                    "member_count": len(active_members),
                    "individual_predictions": {
                        name: pred.detach().cpu().numpy().tolist()
                        for name, pred in ensemble_result.individual_predictions.items()
                    },
                    "member_weights": ensemble_result.member_weights,
                    "consensus_score": ensemble_result.consensus_score,
                    "uncertainty_score": ensemble_result.uncertainty_score,
                },
                processing_time_ms=processing_time_ms,
            )

            return output

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            self.update_inference_metrics(
                (time.time() - start_time) * 1000, success=False
            )
            raise EnsembleException(f"Prediction failed: {e}")

    def predict_async(self, input_data: ModelInput) -> torch.futures.Future:
        """Make asynchronous ensemble prediction."""
        return self.executor.submit(self.predict, input_data)

    def validate_input(self, input_data: ModelInput) -> bool:
        """Validate input data."""
        # Use first active member for validation
        active_members = self.get_active_members()
        if not active_members:
            return False

        first_member = list(active_members.values())[0]
        return first_member.model.validate_input(input_data)

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema from first active member."""
        active_members = self.get_active_members()
        if not active_members:
            return {}

        first_member = list(active_members.values())[0]
        return first_member.model.get_input_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema (ensemble-specific)."""
        return {
            "type": "ensemble_output",
            "predictions": {
                "type": "tensor",
                "description": "Combined ensemble predictions",
            },
            "confidence": {
                "type": "tensor",
                "description": "Ensemble confidence scores",
            },
            "metadata": {
                "ensemble_method": str,
                "member_count": int,
                "individual_predictions": dict,
                "member_weights": dict,
                "consensus_score": float,
                "uncertainty_score": float,
            },
        }

    def train_meta_learner(
        self,
        training_data: List[Tuple[ModelInput, torch.Tensor]],
        validation_split: float = 0.2,
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train meta-learner for stacking ensemble.

        This method trains a secondary model that learns to combine
        the predictions from the base models.
        """
        if self.ensemble_method != EnsembleMethod.STACKING:
            raise EnsembleException(
                "Meta-learner training only available for stacking ensemble"
            )

        self.logger.info("Training meta-learner for stacking ensemble")

        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]

        # Generate base model predictions for training data
        train_features, train_targets = self._generate_meta_features(train_data)
        val_features, val_targets = self._generate_meta_features(val_data)

        # Create meta-learner architecture
        input_dim = train_features.shape[1]
        output_dim = train_targets.shape[1] if len(train_targets.shape) > 1 else 1

        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = nn.MSELoss() if output_dim > 1 else nn.BCEWithLogitsLoss()

        # Training loop
        training_history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.meta_learner.train()
            train_loss = 0.0

            optimizer.zero_grad()
            train_pred = self.meta_learner(train_features)
            loss = criterion(train_pred, train_targets)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # Validation
            self.meta_learner.eval()
            with torch.no_grad():
                val_pred = self.meta_learner(val_features)
                val_loss = criterion(val_pred, val_targets).item()

            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

        self.logger.info("Meta-learner training completed")
        return training_history

    def update_adaptive_weights(
        self, validation_data: List[Tuple[ModelInput, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Update member weights based on recent performance.

        Args:
            validation_data: Recent validation data for weight calculation

        Returns:
            Updated weights for each member
        """
        if not self.adaptive_weights:
            return {name: member.weight for name, member in self.members.items()}

        self.logger.info("Updating adaptive weights based on recent performance")

        # Calculate performance for each member
        member_scores = {}
        active_members = self.get_active_members()

        for name, member in active_members.items():
            predictions = []
            targets = []

            for input_data, target in validation_data:
                try:
                    pred = member.model.predict(input_data)
                    predictions.append(pred.predictions.detach().cpu().numpy())
                    targets.append(target.detach().cpu().numpy())
                except Exception as e:
                    self.logger.warning(f"Error getting prediction from {name}: {e}")
                    continue

            if predictions:
                predictions = np.array(predictions)
                targets = np.array(targets)

                # Calculate accuracy (simple metric, can be enhanced)
                if predictions.shape == targets.shape:
                    score = accuracy_score(
                        targets.flatten(), predictions.flatten() > 0.5
                    )
                    member_scores[name] = score
                    member.performance_score = score

        # Update weights based on performance
        if member_scores:
            total_score = sum(member_scores.values())
            if total_score > 0:
                for name, score in member_scores.items():
                    new_weight = score / total_score * len(member_scores)  # Normalize
                    self.set_member_weight(name, new_weight)

        updated_weights = {name: member.weight for name, member in self.members.items()}
        self.logger.info(f"Updated weights: {updated_weights}")

        return updated_weights

    def calculate_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble members."""
        # This would require validation data to compute actual diversity
        # For now, return placeholder metrics
        return {
            "member_count": len(self.get_active_members()),
            "weight_variance": np.var([m.weight for m in self.members.values()]),
            "average_agreement": self.ensemble_metrics["member_agreement_score"],
        }

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""
        active_members = self.get_active_members()

        return {
            "ensemble_method": self.ensemble_method.value,
            "voting_type": (
                self.voting_type.value if hasattr(self, "voting_type") else None
            ),
            "total_members": len(self.members),
            "active_members": len(active_members),
            "member_details": {
                name: {
                    "weight": member.weight,
                    "is_active": member.is_active,
                    "performance_score": member.performance_score,
                    "model_type": type(member.model).__name__,
                }
                for name, member in self.members.items()
            },
            "ensemble_metrics": self.ensemble_metrics.copy(),
            "adaptive_weights": self.adaptive_weights,
            "has_meta_learner": self.meta_learner is not None,
        }

    # Private methods

    def _get_individual_predictions(
        self, active_members: Dict[str, EnsembleMember], input_data: ModelInput
    ) -> Dict[str, ModelOutput]:
        """Get predictions from all active members."""
        results = {}

        if len(active_members) == 1:
            # Single member - no need for parallelization
            name, member = list(active_members.items())[0]
            results[name] = member.model.predict(input_data)
        else:
            # Multiple members - use parallel prediction
            futures = {}

            for name, member in active_members.items():
                future = self.executor.submit(member.model.predict, input_data)
                futures[name] = future

            # Collect results
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=30)  # 30 second timeout
                except Exception as e:
                    self.logger.error(f"Prediction failed for member {name}: {e}")
                    # Deactivate problematic member
                    self.set_member_active(name, False)

        return results

    def _combine_predictions(
        self,
        individual_results: Dict[str, ModelOutput],
        active_members: Dict[str, EnsembleMember],
    ) -> EnsembleResult:
        """Combine individual predictions based on ensemble method."""
        if not individual_results:
            raise EnsembleException(
                "No individual predictions available for combination"
            )

        # Extract predictions and confidences
        predictions = {}
        confidences = {}
        weights = {}

        for name, result in individual_results.items():
            predictions[name] = result.predictions
            confidences[name] = (
                result.confidence
                if result.confidence is not None
                else torch.ones_like(result.predictions)
            )
            weights[name] = active_members[name].weight

        # Combine based on method
        if self.ensemble_method == EnsembleMethod.VOTING:
            final_pred = self._voting_combine(predictions, weights)
        elif self.ensemble_method == EnsembleMethod.AVERAGING:
            final_pred = self._averaging_combine(predictions)
        elif self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            final_pred = self._weighted_average_combine(predictions, weights)
        elif self.ensemble_method == EnsembleMethod.STACKING:
            final_pred = self._stacking_combine(predictions)
        elif self.ensemble_method == EnsembleMethod.MAX_CONFIDENCE:
            final_pred = self._max_confidence_combine(predictions, confidences)
        else:
            # Default to weighted average
            final_pred = self._weighted_average_combine(predictions, weights)

        # Calculate consensus and uncertainty scores
        consensus_score = self._calculate_consensus_score(predictions)
        uncertainty_score = self._calculate_uncertainty_score(predictions, confidences)

        return EnsembleResult(
            final_prediction=final_pred,
            individual_predictions=predictions,
            individual_confidences=confidences,
            member_weights=weights,
            combination_method=self.ensemble_method.value,
            consensus_score=consensus_score,
            uncertainty_score=uncertainty_score,
        )

    def _voting_combine(
        self, predictions: Dict[str, torch.Tensor], weights: Dict[str, float]
    ) -> torch.Tensor:
        """Combine predictions using voting."""
        if self.voting_type == VotingType.HARD:
            # Hard voting - use predicted classes
            voted_predictions = []
            for name, pred in predictions.items():
                # Convert to class predictions (assuming binary classification for simplicity)
                class_pred = (pred > 0.5).float()
                voted_predictions.append(class_pred * weights[name])

            # Sum weighted votes
            total_votes = torch.stack(voted_predictions).sum(dim=0)
            total_weight = sum(weights.values())

            return (total_votes > total_weight / 2).float()

        else:
            # Soft voting - use probabilities/scores
            weighted_preds = []
            for name, pred in predictions.items():
                weighted_preds.append(pred * weights[name])

            total_weight = sum(weights.values())
            return torch.stack(weighted_preds).sum(dim=0) / total_weight

    def _averaging_combine(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine predictions using simple averaging."""
        pred_stack = torch.stack(list(predictions.values()))
        return torch.mean(pred_stack, dim=0)

    def _weighted_average_combine(
        self, predictions: Dict[str, torch.Tensor], weights: Dict[str, float]
    ) -> torch.Tensor:
        """Combine predictions using weighted averaging."""
        weighted_preds = []
        for name, pred in predictions.items():
            weighted_preds.append(pred * weights[name])

        total_weight = sum(weights.values())
        return torch.stack(weighted_preds).sum(dim=0) / total_weight

    def _stacking_combine(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine predictions using meta-learner."""
        if self.meta_learner is None:
            raise EnsembleException("Meta-learner not trained for stacking ensemble")

        # Stack predictions as features for meta-learner
        pred_features = torch.cat(list(predictions.values()), dim=-1)

        self.meta_learner.eval()
        with torch.no_grad():
            return self.meta_learner(pred_features)

    def _max_confidence_combine(
        self, predictions: Dict[str, torch.Tensor], confidences: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Select prediction with maximum confidence."""
        max_confidence_idx = None
        max_confidence = -1

        for i, (name, conf) in enumerate(confidences.items()):
            mean_conf = torch.mean(conf).item()
            if mean_conf > max_confidence:
                max_confidence = mean_conf
                max_confidence_idx = name

        return predictions[max_confidence_idx]

    def _calculate_consensus_score(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Calculate consensus score among predictions."""
        if len(predictions) < 2:
            return 1.0

        pred_list = list(predictions.values())
        pairwise_agreements = []

        for i in range(len(pred_list)):
            for j in range(i + 1, len(pred_list)):
                # Calculate agreement (can be enhanced based on prediction type)
                agreement = torch.mean((pred_list[i] - pred_list[j]) ** 2).item()
                pairwise_agreements.append(
                    1.0 / (1.0 + agreement)
                )  # Convert to agreement score

        return float(np.mean(pairwise_agreements)) if pairwise_agreements else 0.0

    def _calculate_uncertainty_score(
        self, predictions: Dict[str, torch.Tensor], confidences: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate uncertainty score for the ensemble."""
        # Simple uncertainty based on prediction variance
        pred_stack = torch.stack(list(predictions.values()))
        variance = torch.var(pred_stack, dim=0)
        return float(torch.mean(variance).item())

    def _calculate_ensemble_confidence(
        self, ensemble_result: EnsembleResult
    ) -> torch.Tensor:
        """Calculate ensemble confidence score."""
        # Use inverse of uncertainty as confidence
        uncertainty = ensemble_result.uncertainty_score or 0.0
        base_confidence = 1.0 / (1.0 + uncertainty)

        # Adjust by consensus
        consensus_bonus = ensemble_result.consensus_score or 0.0
        final_confidence = base_confidence * (0.5 + 0.5 * consensus_bonus)

        return torch.full_like(ensemble_result.final_prediction, final_confidence)

    def _generate_meta_features(
        self, data: List[Tuple[ModelInput, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate features for meta-learner training."""
        features_list = []
        targets_list = []

        active_members = self.get_active_members()

        for input_data, target in data:
            member_predictions = []

            for name, member in active_members.items():
                try:
                    pred = member.model.predict(input_data)
                    member_predictions.append(pred.predictions.detach())
                except Exception as e:
                    self.logger.warning(f"Error getting meta-feature from {name}: {e}")
                    # Use zero prediction as fallback
                    member_predictions.append(torch.zeros_like(target))

            if member_predictions:
                # Stack member predictions as features
                features = torch.cat(member_predictions, dim=-1)
                features_list.append(features)
                targets_list.append(target)

        return torch.stack(features_list), torch.stack(targets_list)

    def _update_ensemble_metrics(self, ensemble_result: EnsembleResult) -> None:
        """Update ensemble-specific metrics."""
        self.ensemble_metrics["total_predictions"] += 1

        if ensemble_result.consensus_score and ensemble_result.consensus_score > 0.8:
            self.ensemble_metrics["consensus_predictions"] += 1

        if (
            ensemble_result.uncertainty_score
            and ensemble_result.uncertainty_score > 0.5
        ):
            self.ensemble_metrics["high_uncertainty_predictions"] += 1

        # Update running average of agreement score
        if ensemble_result.consensus_score:
            current_avg = self.ensemble_metrics["member_agreement_score"]
            total_preds = self.ensemble_metrics["total_predictions"]
            self.ensemble_metrics["member_agreement_score"] = (
                current_avg * (total_preds - 1) + ensemble_result.consensus_score
            ) / total_preds
