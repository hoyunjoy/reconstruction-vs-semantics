import torch

from models.reconstruction_decoder import MatchedTokenDecoder


def test_matched_decoder_shape_range_and_backward():
    model = MatchedTokenDecoder(
        image_size=28, patch_size=7, hidden_channels=32, latent_dim=24,
        num_res_blocks=1, norm_groups=8,
    )
    tokens = torch.randn(2, 16, 24, requires_grad=True)
    images = model(tokens)
    assert images.shape == (2, 3, 28, 28)
    assert images.min() >= -1 and images.max() <= 1
    images.square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
