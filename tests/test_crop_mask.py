"""``CropMask`` against the full-frame operations it replaces (TODO.md N4).

Every comparison is exact: a crop must give what the same NumPy or OpenCV call on the full
frame gives, bit for bit, or a stage output would change.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.models.hosted import SamRecord, decode_sam_raster, place_sam_raster
from video2sim.stages.segmentation import CropMask, as_crop
from video2sim.stages.tracking import Track, sam_crop

BoolMask = NDArray[np.bool_]


def _masks(shape: tuple[int, int], seed: int) -> list[BoolMask]:
    """Blobs, specks, edge-touching and empty masks on a frame of ``shape``."""
    rng = np.random.default_rng(seed)
    h, w = shape
    out: list[BoolMask] = [np.zeros(shape, dtype=bool), np.ones(shape, dtype=bool)]
    for _ in range(12):
        m = np.zeros(shape, dtype=bool)
        y0, x0 = int(rng.integers(0, h - 1)), int(rng.integers(0, w - 1))
        y1, x1 = int(rng.integers(y0 + 1, h + 1)), int(rng.integers(x0 + 1, w + 1))
        m[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0)) < rng.uniform(0.2, 1.0)
        out.append(m)
    speck = np.zeros(shape, dtype=bool)
    speck[h - 1, w - 1] = True
    out.append(speck)
    edge = np.zeros(shape, dtype=bool)
    edge[: h // 3, :3] = True
    out.append(edge)
    return out


SHAPES = [(48, 64), (54, 96), (45, 61)]  # (45, 61) divides by no factor


@pytest.mark.parametrize("shape", SHAPES)
def test_a_crop_is_its_full_mask(shape: tuple[int, int]) -> None:
    for full in _masks(shape, 1):
        crop = CropMask.from_full(full)
        assert np.array_equal(crop.full(), full)
        assert crop.shape == full.shape
        assert crop.area == int(full.sum())
        assert crop.any() == bool(full.any())
        rows, cols = np.nonzero(full)
        if rows.size:
            ys, xs = np.sort(rows), np.sort(cols)
            assert crop.bounds == (int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1]))
        else:
            assert crop.bounds is None


@pytest.mark.parametrize("shape", SHAPES)
def test_windows_and_lookups_match_the_full_frame(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(2)
    h, w = shape
    for full in _masks(shape, 3):
        crop = CropMask.from_full(full)
        for _ in range(40):
            r0, r1 = (int(v) for v in rng.integers(-h - 5, h + 5, 2))
            c0, c1 = (int(v) for v in rng.integers(-w - 5, w + 5, 2))
            assert np.array_equal(crop.window(slice(r0, r1), slice(c0, c1)), full[r0:r1, c0:c1])
        assert np.array_equal(crop.window(slice(None), slice(None)), full)
        rows = rng.integers(0, h, 500)
        cols = rng.integers(0, w, 500)
        assert np.array_equal(crop.at(rows, cols), full[rows, cols])


def test_overlap_counts_shared_pixels() -> None:
    masks = _masks((48, 64), 4)
    for a in masks:
        for b in masks:
            got = CropMask.from_full(a).overlap(CropMask.from_full(b))
            assert got == int(np.logical_and(a, b).sum())
    other = CropMask.from_full(np.ones((10, 10), dtype=bool))
    assert CropMask.from_full(masks[1]).overlap(other) == 0


def _resize(full: BoolMask, factor: int, interpolation: int, dsize: Any) -> BoolMask:
    if dsize is None:
        out = cv2.resize(
            full.astype(np.uint8), None, fx=1.0 / factor, fy=1.0 / factor,
            interpolation=interpolation,
        )  # fmt: skip
    else:
        out = cv2.resize(full.astype(np.uint8), dsize, interpolation=interpolation)
    result: BoolMask = np.asarray(out) != 0
    return result


@pytest.mark.parametrize("shape", [*SHAPES, (1080, 1920)])
@pytest.mark.parametrize("factor", [1, 2, 3, 4])
@pytest.mark.parametrize("interpolation", [cv2.INTER_AREA, cv2.INTER_NEAREST])
def test_downscaling_matches_resizing_the_full_frame(
    shape: tuple[int, int], factor: int, interpolation: int
) -> None:
    masks = _masks(shape, 5)[:6] if shape[0] > 500 else _masks(shape, 5)
    for full in masks:
        crop = CropMask.from_full(full)
        small = crop.downscaled(factor, interpolation)
        assert np.array_equal(small.full(), _resize(full, factor, interpolation, None))
        dsize = (shape[1] // factor, shape[0] // factor)
        sized = crop.downscaled(factor, interpolation, dsize)
        assert np.array_equal(sized.full(), _resize(full, factor, interpolation, dsize))


def test_masks_at_odd_offsets_downscale_the_same() -> None:
    """A box that starts and ends off the block grid still shrinks like the frame."""
    rng = np.random.default_rng(6)
    for _ in range(200):
        full = np.zeros((96, 128), dtype=bool)
        y0, x0 = int(rng.integers(0, 90)), int(rng.integers(0, 120))
        full[y0 : y0 + int(rng.integers(1, 7)), x0 : x0 + int(rng.integers(1, 9))] = True
        crop = CropMask.from_full(full)
        for factor in (2, 4):
            for interp in (cv2.INTER_AREA, cv2.INTER_NEAREST):
                got = crop.downscaled(factor, interp).full()
                assert np.array_equal(got, _resize(full, factor, interp, None))


def test_a_crop_outside_its_frame_is_refused() -> None:
    with pytest.raises(ValueError):
        CropMask(5, 0, np.ones((6, 2), dtype=bool), (10, 10))
    with pytest.raises(ValueError):
        CropMask(0, 0, np.ones((2, 2), dtype=np.uint8), (10, 10))


def test_a_track_cuts_full_masks_to_crops() -> None:
    full = _masks((48, 64), 7)[3]
    track = Track(4, {2: full, 3: CropMask.from_full(full)})
    assert all(isinstance(m, CropMask) for m in track.masks.values())
    assert np.array_equal(track.masks[2].full(), full)
    assert as_crop(track.masks[3]) is track.masks[3]


def _record(box: tuple[float, float, float, float], frame: tuple[int, int]) -> SamRecord:
    from meta_sam_parser._mask_codec import _encode_raster

    rows, cols = 7, 9
    raster = (np.random.default_rng(8).random((rows, cols)) < 0.5).astype(np.uint8)
    payload = _encode_raster(raster.tobytes(), cols, rows)
    return SamRecord(0, 1, box, frame, (0, 0), (rows, cols), payload)


def test_sam_crops_match_the_placed_masks() -> None:
    for box in [(3.0, 4.0, 40.0, 30.0), (0.0, 0.0, 63.0, 47.0), (10.6, 2.2, 11.4, 2.9)]:
        record = _record(box, (64, 48))
        raster = decode_sam_raster(record)
        assert np.array_equal(sam_crop(record, raster).full(), place_sam_raster(record, raster))
    wide = _record((50.0, 4.0, 70.0, 30.0), (64, 48))  # leaves the frame: the old error
    raster = decode_sam_raster(wide)
    with pytest.raises(ValueError):
        place_sam_raster(wide, raster)
    with pytest.raises(ValueError):
        sam_crop(wide, raster)


# --- N7: decoded rasters stored per answer ----------------------------------------------


def _answer(records: int) -> str:
    """A SAM 3.1 answer with ``records`` lines of one object each."""
    from meta_sam_parser._mask_codec import _encode_raster

    rng = np.random.default_rng(9)
    lines = []
    for i in range(records):
        rows, cols = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        raster = (rng.random((rows, cols)) < 0.5).astype(np.uint8)
        payload = _encode_raster(raster.tobytes(), cols, rows)
        lines.append(
            f"<{i}f>3<|box;x1=2;y1=3;x2=30;y2=20;w=64;h=48|>"
            f"<|mask;x=2;y=3;data={rows},{cols},{payload}|>"
        )
    return "\n".join(lines)


def test_stored_rasters_give_the_same_tracks(tmp_path: Any) -> None:
    from video2sim.models import hosted
    from video2sim.stages.tracking import tracks_from_text

    text = _answer(6)
    frames = list(range(10, 16))
    plain = tracks_from_text(text, frames)
    first = tracks_from_text(text, frames, raster_cache=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    calls = []
    real = hosted.decode_sam_rasters
    hosted.decode_sam_rasters = lambda r, w=None: calls.append(1) or real(r, w)  # type: ignore[assignment,func-returns-value]
    try:
        second = tracks_from_text(text, frames, raster_cache=tmp_path)
    finally:
        hosted.decode_sam_rasters = real
    assert calls == []  # read back, not decoded
    for tracks in (first, second):
        assert [t.track_id for t in tracks] == [t.track_id for t in plain]
        for a, b in zip(tracks, plain, strict=True):
            assert sorted(a.masks) == sorted(b.masks)
            for f in a.masks:
                assert np.array_equal(a.masks[f].full(), b.masks[f].full())


def test_a_stored_file_that_does_not_fit_is_decoded_again(tmp_path: Any) -> None:
    from video2sim.models.hosted import cached_sam_rasters, decode_sam_rasters, parse_sam_text

    text = _answer(4)
    records = parse_sam_text(text)
    cached_sam_rasters(text, records, tmp_path)
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"not an npz")
    again = cached_sam_rasters(text, records, tmp_path)
    expected = decode_sam_rasters(records)
    assert all(np.array_equal(a, b) for a, b in zip(again, expected, strict=True))
    from video2sim.models.hosted import _read_rasters

    assert _read_rasters(path, records) is not None  # rewritten by the second call
    other = parse_sam_text(_answer(3))  # three records against a file for four
    assert _read_rasters(path, other) is None
