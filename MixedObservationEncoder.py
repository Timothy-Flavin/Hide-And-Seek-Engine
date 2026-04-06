import numpy as np
import torch
import torch.nn as nn


def infer_encoder_out_dim(encoder: nn.Module, input_dim: int) -> int:
    if hasattr(encoder, "output_dim"):
        return int(getattr(encoder, "output_dim"))
    with torch.no_grad():
        dummy = torch.zeros(1, input_dim, dtype=torch.float32)
        out = encoder(dummy)
    if out.ndim < 2:
        raise ValueError("Custom encoder must output [B, F]")
    return int(out.shape[-1])


class ReshapeToSpatial(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

class MixedObservationEncoder(nn.Module):
    """Encodes flattened [spatial | vector] observations into a dense feature vector."""

    def __init__(
        self,
        spatial_shape,
        vector_dim: int,
        spatial_hidden_dim: int = 128,
        vector_hidden_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.vector_dim = int(vector_dim)
        self.spatial_dim = int(np.prod(self.spatial_shape))
        self.output_dim = int(output_dim)

        if len(self.spatial_shape) == 3:
            c, h, w = self.spatial_shape
            flattened_size = 64 * (h // 4) * (w // 4)
            self.spatial_encoder = nn.Sequential(
                ReshapeToSpatial((c, h, w)),
                nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(flattened_size, spatial_hidden_dim),
                nn.ReLU(),
            )
        elif len(self.spatial_shape) == 2:
            h, w = self.spatial_shape
            flattened_size = 64 * (h // 4) * (w // 4)
            self.spatial_encoder = nn.Sequential(
                ReshapeToSpatial((1, h, w)),
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(flattened_size, spatial_hidden_dim),
                nn.ReLU(),
            )
        else:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(self.spatial_dim, spatial_hidden_dim),
                nn.ReLU(),
                nn.Linear(spatial_hidden_dim, spatial_hidden_dim),
                nn.ReLU(),
            )

        if self.vector_dim > 0:
            self.vector_encoder = nn.Sequential(
                nn.Linear(self.vector_dim, vector_hidden_dim),
                nn.ReLU(),
                nn.Linear(vector_hidden_dim, vector_hidden_dim),
                nn.ReLU(),
            )
            fusion_in_dim = spatial_hidden_dim + vector_hidden_dim
            self.has_vector = True
        else:
            self.vector_encoder = None
            fusion_in_dim = spatial_hidden_dim
            self.has_vector = False

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_flat = x[..., : self.spatial_dim]
        spatial_feat = self.spatial_encoder(spatial_flat)
        if self.has_vector:
            vector = x[..., self.spatial_dim :]
            vector_feat = self.vector_encoder(vector)
            fused = torch.cat([spatial_feat, vector_feat], dim=-1)
        else:
            fused = spatial_feat
        return self.fusion(fused)
