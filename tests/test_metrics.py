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
