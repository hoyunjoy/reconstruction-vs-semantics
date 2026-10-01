import numpy as np
import pytest

from datasets.pusht_video_frames import select_frame_indices


def test_random_frame_selection_is_reproducible_unique_and_epoch_dependent():
    arguments = {
        "sequence_length": 100,
        "frame_count": 16,
        "strategy": "random",
        "seed": 42,
        "source_index": 17,
    }
    first = select_frame_indices(epoch=0, **arguments)
    repeated = select_frame_indices(epoch=0, **arguments)
    next_epoch = select_frame_indices(epoch=1, **arguments)

    np.testing.assert_array_equal(first, repeated)
    assert len(first) == len(np.unique(first)) == 16
    assert np.all(first[:-1] < first[1:])
    assert first.min() >= 0 and first.max() < 100
    assert not np.array_equal(first, next_epoch)


def test_uniform_and_all_frame_selection():
    uniform = select_frame_indices(
        sequence_length=10,
        frame_count=4,
        strategy="uniform",
        seed=42,
        epoch=0,
        source_index=0,
    )
    all_frames = select_frame_indices(
        sequence_length=5,
        frame_count=-1,
        strategy="uniform",
        seed=42,
        epoch=0,
        source_index=0,
    )
    np.testing.assert_array_equal(uniform, np.array([0, 3, 6, 9]))
    np.testing.assert_array_equal(all_frames, np.arange(5))


def test_invalid_frame_selection_arguments():
    with pytest.raises(ValueError):
        select_frame_indices(
            sequence_length=10,
            frame_count=0,
            strategy="random",
            seed=42,
            epoch=0,
            source_index=0,
        )
