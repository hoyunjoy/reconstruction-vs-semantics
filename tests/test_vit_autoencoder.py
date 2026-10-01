import torch
from torch import nn

from models.vit_autoencoder import ReconstructionDecoder, ViTAutoencoder


class FakeViTS14(nn.Module):
    patch_size = 14
    num_features = 384

    def forward_features(self, images):
        batch = images.shape[0]
        return {"x_norm_patchtokens": torch.zeros(batch, 256, 384)}


class FakeScratchEncoder(nn.Module):
    emb_dim = 384
    grid_size = 16
    patch_size = 14

    def forward(self, images):
        return torch.zeros(images.shape[0], 256, 384, device=images.device)


def test_existing_decoder_parameter_count_and_shape():
    decoder = ReconstructionDecoder()
    assert sum(p.numel() for p in decoder.parameters()) == 2_611_543
    output = decoder(torch.randn(2, 256, 384))
    assert output.shape == (2, 3, 224, 224)
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_autoencoder_shape_without_network(monkeypatch):
    monkeypatch.setattr(
        "models.vit_autoencoder.DinoV2ScratchEncoder",
        lambda **_: FakeScratchEncoder(),
    )
    model = ViTAutoencoder()
    reconstruction, tokens = model(torch.randn(2, 3, 224, 224), return_latent=True)
    assert tokens.shape == (2, 256, 384)
    assert reconstruction.shape == (2, 3, 224, 224)
