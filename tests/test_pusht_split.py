import pickle

import torch

from scripts.create_pusht_split import (
    REQUIRED_TENSORS,
    build_manifest,
    load_metadata_counts,
    render_manifest,
    validate_manifest,
)


def make_fake_pusht(root, candidate_count: int) -> None:
    train_dir = root / "train"
    video_dir = train_dir / "obses"
    video_dir.mkdir(parents=True)
    with (train_dir / "seq_lengths.pkl").open("wb") as handle:
        pickle.dump([5] * candidate_count, handle)
    for filename in REQUIRED_TENSORS:
        torch.save(torch.zeros(candidate_count, 5, 2), train_dir / filename)
    for index in range(candidate_count):
        (video_dir / f"episode_{index:03d}.mp4").touch()


def test_manifest_is_deterministic_valid_and_disjoint(tmp_path):
    candidate_count = 20
    make_fake_pusht(tmp_path, candidate_count)
    metadata_counts = load_metadata_counts(tmp_path / "train")
    arguments = {
        "candidate_count": candidate_count,
        "seed": 42,
        "train_count": 8,
        "validation_count": 1,
        "test_count": 1,
    }

    first = build_manifest(**arguments)
    second = build_manifest(**arguments)
    validate_manifest(first, tmp_path, metadata_counts)

    assert render_manifest(first) == render_manifest(second)
    assert first["dataset"]["source_split"] == "train"
    assert first["dataset"]["official_val_used"] is False
    assert first["sampling"]["algorithm"] == (
        "numpy.random.Generator(numpy.random.PCG64).permutation"
    )
    assert first["splits"]["train"] == [15, 9, 14, 7, 12, 10, 6, 19]
    assert first["splits"]["validation"] == [3]
    assert first["splits"]["test"] == [0]
    assert {name: len(values) for name, values in first["splits"].items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    selected = [value for values in first["splits"].values() for value in values]
    assert len(selected) == len(set(selected)) == 10
    assert min(selected) >= 0
    assert max(selected) < candidate_count


def test_manifest_validation_rejects_a_missing_video(tmp_path):
    make_fake_pusht(tmp_path, 10)
    metadata_counts = load_metadata_counts(tmp_path / "train")
    manifest = build_manifest(
        candidate_count=10,
        seed=42,
        train_count=3,
        validation_count=1,
        test_count=1,
    )
    selected_index = manifest["splits"]["train"][0]
    (tmp_path / "train" / "obses" / f"episode_{selected_index:03d}.mp4").unlink()

    try:
        validate_manifest(manifest, tmp_path, metadata_counts)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing selected MP4 was not detected")
