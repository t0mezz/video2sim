from video2sim.evaluation.ground_truth import size_convention_errors, yaw_is_defined


def test_tall_object_is_valid() -> None:
    # A bottle: height is the up-axis and exceeds the horizontal sides.
    assert size_convention_errors([0.093, 0.093, 0.246]) == []


def test_flat_object_is_valid() -> None:
    assert size_convention_errors([0.280, 0.280, 0.018]) == []


def test_length_below_width_is_rejected() -> None:
    assert size_convention_errors([0.150, 0.190, 0.160]) != []


def test_wrong_arity_and_non_positive_are_rejected() -> None:
    assert size_convention_errors([0.1, 0.1]) != []
    assert size_convention_errors([0.1, 0.1, 0.0]) != []
    assert size_convention_errors([0.1, 0.1, float("nan")]) != []


def test_yaw_undefined_for_symmetric_footprint() -> None:
    assert not yaw_is_defined([0.070, 0.070, 0.070])  # cube
    assert not yaw_is_defined([0.067, 0.068, 0.240])  # spray can, 1 mm caliper spread
    assert yaw_is_defined([0.190, 0.150, 0.160])  # monkey
