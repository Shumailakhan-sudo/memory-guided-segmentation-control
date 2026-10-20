import numpy as np
import pytest

from memory_guided_vos import preprocessing as pp
from memory_guided_vos import prompting


def test_alignment_identity_at_24fps():
    timestamps = np.arange(10) / 24.0
    indices = pp.align_to_reference_rate(timestamps, 24.0)

    assert np.array_equal(indices, np.arange(10))
    assert pp.alignment_report(indices, 10)["is_identity"]


def test_alignment_upsampling_repeats_frames():
    timestamps = np.arange(5) / 12.0
    report = pp.alignment_report(pp.align_to_reference_rate(timestamps, 24.0), 5)

    assert report["n_repeated_source_frames"] > 0


def test_alignment_tie_goes_to_earlier_frame():
    timestamps = np.array([0.0, 2 / 24.0])
    indices = pp.align_to_reference_rate(timestamps, 24.0, n_out=2)

    assert indices[1] == 0


def test_short_side_scale():
    scale, dimensions = pp.short_side_scale(1080, 1920)

    assert scale == pytest.approx(480 / 1080)
    assert dimensions == (480, 853)
    assert pp.short_side_scale(480, 854)[1] == (480, 854)


def test_mask_resize_preserves_labels():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:20, 5:20] = 1
    mask[25:35, 30:50] = 2
    resized = pp.resize_mask(mask, (20, 30))

    assert set(np.unique(resized)) <= {0, 1, 2}


def test_rgb_resize_shape():
    image = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)

    assert pp.resize_rgb(image, (20, 30)).shape == (20, 30, 3)


def test_pad_to_multiple():
    image = np.zeros((481, 853, 3))
    padded, padding = pp.pad_to_multiple(image, 8)
    top, bottom, left, right = padding

    assert padded.shape[0] % 8 == 0
    assert padded.shape[1] % 8 == 0
    assert padded.shape[0] - top - bottom == 481
    assert padded.shape[1] - left - right == 853


def test_bbox_prompt(square_mask):
    mask = square_mask(30, 30, 5, 7, 10)

    assert prompting.bbox_prompt(mask) == (7, 5, 16, 14)


def test_center_point_is_interior_maximum(square_mask):
    mask = square_mask(31, 31, 5, 5, 21)

    assert prompting.center_point_prompt(mask) == (15, 15)


def test_center_point_object_touching_border(square_mask):
    mask = square_mask(20, 20, 0, 0, 9)
    x, y = prompting.center_point_prompt(mask)

    assert mask[y, x]


def test_none_prompt_is_not_guessed(square_mask):
    with pytest.raises(prompting.UndefinedPromptError):
        prompting.make_prompt("none", square_mask(10, 10, 2, 2, 3))


def test_first_valid_reference():
    masks = [np.zeros((3, 3), dtype=bool), np.eye(3, dtype=bool)]

    assert prompting.first_valid_reference(masks) == 1
