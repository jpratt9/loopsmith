"""Tests for the loopsmith module."""

from unittest.mock import patch

import numpy as np
import pytest

from loopsmith import (
    compute_ncc_matrix,
    find_best_for_target,
    find_best_loop,
    find_top_loops,
    analyze_video,
)


def _make_frame(value, size=100):
    """Create a normalized flat frame with a known pattern."""
    frame = np.random.RandomState(value).randn(size).astype(np.float32)
    frame = (frame - frame.mean()) / frame.std()
    return frame


class TestComputeNccMatrix:
    def test_self_similarity_is_one(self):
        rows = [_make_frame(1), _make_frame(2)]
        ncc_matrix, gaps = compute_ncc_matrix(rows, [0, 10])
        assert ncc_matrix[0, 0] == pytest.approx(1.0, abs=0.01)
        assert ncc_matrix[1, 1] == pytest.approx(1.0, abs=0.01)

    def test_gaps_computed_correctly(self):
        rows = [_make_frame(1), _make_frame(2), _make_frame(3)]
        ncc_matrix, gaps = compute_ncc_matrix(rows, [0, 30, 90])
        assert gaps[0, 1] == 30
        assert gaps[0, 2] == 90
        assert gaps[1, 2] == 60
        assert gaps[1, 0] == -30

    def test_identical_frames_have_high_ncc(self):
        frame = _make_frame(42)
        rows = [frame.copy(), frame.copy()]
        ncc_matrix, _ = compute_ncc_matrix(rows, [0, 100])
        assert ncc_matrix[0, 1] == pytest.approx(1.0, abs=0.01)

    def test_different_frames_have_low_ncc(self):
        rows = [_make_frame(1), _make_frame(999)]
        ncc_matrix, _ = compute_ncc_matrix(rows, [0, 100])
        assert ncc_matrix[0, 1] < 0.5


class TestFindBestLoop:
    def test_finds_loop_above_threshold(self):
        frame = _make_frame(42)
        rows = [frame.copy(), _make_frame(2), frame.copy()]
        indices = [0, 30, 90]
        result = find_best_loop(rows, indices, threshold=0.90)
        assert result is not None
        start, end, ncc = result
        assert start == 0
        assert end == 90
        assert ncc > 0.90

    def test_returns_none_below_threshold(self):
        rows = [_make_frame(1), _make_frame(2), _make_frame(3)]
        indices = [0, 30, 60]
        result = find_best_loop(rows, indices, threshold=0.99)
        assert result is None

    def test_returns_none_with_fewer_than_two_frames(self):
        result = find_best_loop([_make_frame(1)], [0], threshold=0.5)
        assert result is None

    def test_picks_largest_gap(self):
        frame = _make_frame(42)
        rows = [frame.copy(), frame.copy(), _make_frame(99), frame.copy()]
        indices = [0, 10, 50, 100]
        result = find_best_loop(rows, indices, threshold=0.90)
        assert result is not None
        start, end, ncc = result
        # Should pick f0->f100 (gap=100) over f0->f10 (gap=10)
        assert end - start == 100


class TestFindBestForTarget:
    def test_finds_closest_to_target(self):
        frame = _make_frame(42)
        # 3 identical frames at 0, 150, 300 -> gaps of 5s and 10s at 30fps
        rows = [frame.copy(), frame.copy(), frame.copy()]
        indices = [0, 150, 300]
        result = find_best_for_target(rows, indices, fps=30.0, target_seconds=9.0)
        assert result is not None
        start, end, dur, ncc = result
        # Should pick the 10s pair (closest to 9s target)
        assert dur == pytest.approx(10.0, abs=0.1)
        assert ncc > 0.99

    def test_picks_closest_duration_among_top_ncc(self):
        frame = _make_frame(42)
        # 4 identical frames at different gaps: 2s, 5s, 8s, 10s at 30fps
        rows = [frame.copy(), frame.copy(), frame.copy(), frame.copy()]
        indices = [0, 60, 150, 300]
        # Target 7s -> should pick 8s (f0->f240 isn't available, but f60->f300 = 8s)
        result = find_best_for_target(rows, indices, fps=30.0, target_seconds=7.0)
        assert result is not None
        # All pairs have ~100% NCC, so it picks purely by closest duration
        assert result[3] > 0.9

    def test_skips_short_gaps(self):
        frame = _make_frame(42)
        # Two identical frames only 10 frames apart at 30fps = 0.33s (< 1s threshold)
        rows = [frame.copy(), frame.copy()]
        indices = [0, 10]
        result = find_best_for_target(rows, indices, fps=30.0, target_seconds=5.0)
        assert result is None

    def test_returns_none_with_fewer_than_two_frames(self):
        result = find_best_for_target([_make_frame(1)], [0], fps=30.0, target_seconds=5.0)
        assert result is None

    def test_returns_none_when_no_high_ncc_pairs(self):
        # Two different frames -- NCC will be low, below min_ncc threshold
        rows = [_make_frame(1), _make_frame(2)]
        indices = [0, 300]
        result = find_best_for_target(rows, indices, fps=30.0, target_seconds=10.0)
        assert result is None


class TestFindTopLoops:
    def test_returns_top_by_ncc(self):
        frame_a = _make_frame(1)
        frame_b = _make_frame(2)
        rows = [frame_a.copy(), frame_b.copy(), frame_a.copy()]
        indices = [0, 30, 60]
        by_ncc, by_gap = find_top_loops(rows, indices, fps=30.0, top_n=5)
        assert len(by_ncc) > 0
        # First entry should have highest NCC
        assert by_ncc[0][3] >= by_ncc[-1][3]

    def test_returns_top_by_gap(self):
        frame = _make_frame(42)
        rows = [frame.copy(), _make_frame(2), frame.copy()]
        indices = [0, 30, 90]
        by_ncc, by_gap = find_top_loops(rows, indices, fps=30.0, top_n=5)
        assert len(by_gap) > 0
        # First entry should have largest gap
        assert by_gap[0][2] >= by_gap[-1][2]

    def test_empty_with_fewer_than_two_frames(self):
        by_ncc, by_gap = find_top_loops([_make_frame(1)], [0], fps=30.0)
        assert by_ncc == []
        assert by_gap == []

    def test_gap_duration_is_in_seconds(self):
        frame = _make_frame(42)
        rows = [frame.copy(), frame.copy()]
        indices = [0, 90]
        by_ncc, _ = find_top_loops(rows, indices, fps=30.0)
        # 90 frames at 30fps = 3.0 seconds
        assert by_ncc[0][2] == pytest.approx(3.0, abs=0.01)


class TestAnalyzeVideo:
    @patch("loopsmith.extract_frames")
    def test_returns_loop_info(self, mock_extract):
        frame = _make_frame(42)
        mock_extract.return_value = (
            [frame.copy(), _make_frame(2), frame.copy()],
            [0, 30, 90],
            30.0,
            100,
        )
        result = analyze_video("/fake/video.mp4", threshold=0.85, downsample=3)
        assert result["file"] == "video.mp4"
        assert result["total_frames"] == 100
        assert result["fps"] == 30.0
        assert result["loop"] is not None
        assert result["loop"]["start_frame"] == 0
        assert result["loop"]["end_frame"] == 90
        assert result["loop"]["ncc"] > 0.90

    @patch("loopsmith.extract_frames")
    def test_returns_none_when_no_loop(self, mock_extract):
        mock_extract.return_value = (
            [_make_frame(1), _make_frame(2), _make_frame(3)],
            [0, 30, 60],
            30.0,
            100,
        )
        result = analyze_video("/fake/video.mp4", threshold=0.99, downsample=3)
        assert result["loop"] is None

    @patch("loopsmith.extract_frames")
    def test_handles_zero_fps(self, mock_extract):
        mock_extract.return_value = ([], [], 0.0, 0)
        result = analyze_video("/fake/video.mp4", threshold=0.85, downsample=3)
        assert result["duration"] == 0
        assert result["loop"] is None
