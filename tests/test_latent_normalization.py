import torch

from metrics.latent_normalization import LatentNormalizer, StreamingChannelMoments


def test_streaming_moments_match_direct_population_statistics():
    generator = torch.Generator().manual_seed(42)
    first = torch.randn(3, 5, 4, generator=generator) * 2 + 1
    second = torch.randn(7, 5, 4, generator=generator) * 0.5 - 3
    combined = torch.cat((first, second), dim=0)

    moments = StreamingChannelMoments(channel_dim=4)
    moments.update(first)
    moments.update(second)
    mean, std, count = moments.finalize(minimum_std=1e-8)

    torch.testing.assert_close(mean, combined.mean(dim=(0, 1)))
    torch.testing.assert_close(
        std, combined.var(dim=(0, 1), correction=0).sqrt()
    )
    assert count == combined.shape[0] * combined.shape[1]


def test_latent_normalizer_produces_unit_stats_and_round_trips():
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(20, 8, 6, generator=generator)
    mean = values.mean(dim=(0, 1))
    std = values.var(dim=(0, 1), correction=0).sqrt()
    normalizer = LatentNormalizer(mean, std)

    normalized = normalizer.normalize(values)
    restored = normalizer.denormalize(normalized)

    torch.testing.assert_close(
        normalized.mean(dim=(0, 1)), torch.zeros(6), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        normalized.var(dim=(0, 1), correction=0),
        torch.ones(6),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(restored, values)
