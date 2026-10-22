"""Public training lifecycle around native DMD algorithms and Program bindings.

Metadata and configuration imports remain independent of Torch and model assets.
"""
from .config import TrainingConfig, load_training_config
from .adapters import InferenceProjection, ModelComponents

__all__ = ["TrainingConfig", "load_training_config", "ModelComponents", "InferenceProjection"]
