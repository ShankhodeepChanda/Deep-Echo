"""
CNN Autoencoder architecture for SETI anomaly detection.

Implements a convolutional encoder-decoder for 64x64 single-channel spectrograms.
Includes configurable latent dimensions, CPU/CUDA compatibility, and model summary.
"""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CNNAutoencoder(nn.Module):
    """Convolutional autoencoder for spectrogram anomaly detection."""

    def __init__(self, input_size: Tuple[int, int, int] = (1, 64, 64), latent_dim: int = 48):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder: (C, H, W) -> (latent_dim,)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 1x64x64 -> 16x32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x32x32 -> 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x16x16 -> 64x8x8
            nn.ReLU(),
            nn.Flatten(),  # -> 64*8*8 = 4096
            nn.Linear(64 * 8 * 8, latent_dim),  # -> latent_dim
            nn.ReLU(),
        )

        # Decoder: (latent_dim) -> (C, H, W)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),  # -> 64*8*8
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),  # -> 64x8x8
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x8x8 -> 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x16x16 -> 16x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x32x32 -> 1x64x64
            nn.Sigmoid(),  # Output in [0, 1] for normalized input
        )

        # Initialize parameters with stable initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize weights with stable initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input spectrogram to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to spectrogram."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode -> decode."""
        z = self.encode(x)
        return self.decode(z)

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Batch encode for feature extraction."""
        return self.encode(x)

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Batch decode for reconstruction."""
        return self.decode(z)

    def summary(self) -> str:
        """Generate a text summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_layers = len([m for m in self.encoder if isinstance(m, (nn.Conv2d, nn.Linear))])
        decoder_layers = len([m for m in self.decoder if isinstance(m, (nn.ConvTranspose2d, nn.Linear))])
        return (
            f"CNNAutoencoder Summary:\n"
            f"  Input: {self.input_size}\n"
            f"  Latent dimension: {self.latent_dim}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
            f"  Encoder layers: {encoder_layers}\n"
            f"  Decoder layers: {decoder_layers}\n"
            f"  Encoder architecture: Conv2d(1->16->32->64) with stride 2, ReLU\n"
            f"  Decoder architecture: ConvTranspose2d(64->32->16->1) with stride 2, ReLU, Sigmoid\n"
        )

    def save_checkpoint(self, path: str) -> None:
        """Save model weights and architecture configuration."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "input_size": self.input_size,
            "latent_dim": self.latent_dim,
            "architecture_summary": self.summary(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[str] = None) -> "CNNAutoencoder":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            input_size=checkpoint.get("input_size", (1, 64, 64)),
            latent_dim=checkpoint.get("latent_dim", 48),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Generate a dictionary summary of model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            layer_info.append({
                "name": name,
                "type": module.__class__.__name__,
                "params": sum(p.numel() for p in module.parameters()),
            })
    return {
        "total_params": total,
        "trainable_params": trainable,
        "layer_count": len(layer_info),
        "layers": layer_info,
    }


def validate_model_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> bool:
    """Validate that model produces correct output shape for input."""
    device = next(model.parameters()).device
    x = torch.randn(1, *input_shape).to(device)
    try:
        with torch.no_grad():
            y = model(x)
        expected_shape = input_shape
        return y.shape == (1, *expected_shape)
    except Exception as e:
        logger.error(f"Model shape validation failed: {e}")
        return False


if __name__ == "__main__":
    # Self-test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CNNAutoencoder(latent_dim=48).to(device)
    print(f"Model initialized on device: {device}")
    print(model.summary())

    # Test forward pass with 64x64 input
    x = torch.randn(4, 1, 64, 64).to(device)  # Batch of 4 samples
    with torch.no_grad():
        reconstructed = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Latent dimension: {model.latent_dim}")

    # Get summary
    summary_dict = get_model_summary(model)
    print(f"\nSummary dict: {summary_dict}")

    # Validate shape
    is_valid = validate_model_shape(model, (1, 64, 64))
    print(f"\nShape validation passed: {is_valid}")

    # Check parameter count
    total_params = summary_dict["total_params"]
    print(f"Total parameters: {total_params:,}")

    print("\nPhase 7 CNN Autoencoder complete!")
