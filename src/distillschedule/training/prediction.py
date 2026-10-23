"""Explicit diffusion prediction conventions, independent of model imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionSpec:
    kind: str

    def __post_init__(self):
        if self.kind not in ('epsilon', 'velocity', 'flow', 'x0'):
            raise ValueError('prediction kind must be epsilon, velocity, flow or x0')

    def to_x0(self, prediction, sample, *, sigma, alpha=None):
        if self.kind == 'x0':
            return prediction
        if self.kind == 'flow':
            # Rectified-flow convention x_t=(1-t)*x0+t*epsilon.
            return sample - sigma * prediction
        if alpha is None:
            raise ValueError('diffusion epsilon/velocity conversion requires alpha')
        if self.kind == 'epsilon':
            return (sample - sigma * prediction) / alpha
        return alpha * sample - sigma * prediction

    def to_epsilon(self, prediction, sample, *, sigma, alpha=None):
        if self.kind == 'epsilon':
            return prediction
        if self.kind == 'velocity':
            if alpha is None:
                raise ValueError('velocity conversion requires alpha')
            return alpha * prediction + sigma * sample
        if self.kind == 'flow':
            return sample + (1 - sigma) * prediction
        if alpha is None:
            raise ValueError('x0 conversion requires alpha')
        return (sample - alpha * prediction) / sigma
