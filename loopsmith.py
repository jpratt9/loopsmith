"""Find the longest seamlessly-loopable segment in a video using NCC.

Uses vectorized N^2 pairwise Normalized Cross-Correlation (a single matrix
multiply) to find the largest frame gap whose endpoints match above a
similarity threshold -- i.e. the longest sub-clip you can loop without a
visible seam.

CLI:
    loopsmith video.mp4
    loopsmith path/to/videos/                 # batch every .mp4/.mov in a dir
    loopsmith video.mp4 --threshold 0.80
    loopsmith video.mp4 --downsample 5
    loopsmith video.mp4 --detail              # top-10 by similarity and length
    loopsmith video.mp4 --detail --target-length 6
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

THUMB = 160  # scale longest edge to this for comparison
DEFAULT_THRESHOLD = 0.85
DEFAULT_DOWNSAMPLE = 3  # every Nth frame


def extract_frames(video_path, downsample):
    """Extract downsampled, normalized grayscale frames from a video.

    Returns:
        Tuple of (list of flattened normalized frames, list of original frame indices, fps).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frame_indices = []
    for i in range(0, total, downsample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = THUMB / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        g = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        std = g.std()
        if std < 1.0:
            continue
        normed = ((g - g.mean()) / std).flatten()
        rows.append(normed)
        frame_indices.append(i)
    cap.release()

    return rows, frame_indices, fps, total


def compute_ncc_matrix(rows, frame_indices):
    """Compute all pairwise NCCs via matrix multiplication.

    Returns:
        Tuple of (ncc_matrix, gaps_matrix, frame_indices array).
    """
    mat = np.stack(rows)  # (N, pixels)
    ncc_matrix = mat @ mat.T / mat.shape[1]  # (N, N)
    idx_arr = np.array(frame_indices)
    gaps = idx_arr[None, :] - idx_arr[:, None]  # (N, N)
    return ncc_matrix, gaps


def find_best_loop(rows, frame_indices, threshold):
    """Find the largest frame gap with NCC above threshold.

    Returns:
        Tuple of (best_start_frame, best_end_frame, best_ncc) or None if no loop found.
    """
    n = len(rows)
    if n < 2:
        return None

    ncc_matrix, gaps = compute_ncc_matrix(rows, frame_indices)
    mask = (ncc_matrix >= threshold) & (gaps > 0)

    if not mask.any():
        return None

    qualified_gaps = np.where(mask, gaps, 0)
    flat_idx = np.argmax(qualified_gaps)
    a, b = divmod(flat_idx, n)

    return frame_indices[a], frame_indices[b], float(ncc_matrix[a, b])


def find_best_for_target(rows, frame_indices, fps, target_seconds, min_ncc=0.90):
    """Find the best loop closest to target_seconds among high-NCC pairs.

    Filters to pairs with NCC >= min_ncc and duration >= 1s, then picks
    the one closest to the target duration.

    Returns:
        Tuple of (start_frame, end_frame, gap_seconds, ncc) or None.
    """
    n = len(rows)
    if n < 2:
        return None

    ncc_matrix, gaps = compute_ncc_matrix(rows, frame_indices)

    mask = gaps > 0
    ai, bi = np.where(mask)
    nccs = ncc_matrix[ai, bi]
    durations = gaps[ai, bi].astype(float) / fps

    # Filter: NCC >= min_ncc AND duration >= 1s
    valid = (nccs >= min_ncc) & (durations >= 1.0)
    if not valid.any():
        return None

    ai, bi, nccs, durations = ai[valid], bi[valid], nccs[valid], durations[valid]

    # Pick closest to target duration
    best_k = np.argmin(np.abs(durations - target_seconds))
    return (
        frame_indices[ai[best_k]],
        frame_indices[bi[best_k]],
        float(durations[best_k]),
        float(nccs[best_k]),
    )


def find_top_loops(rows, frame_indices, fps, top_n=10):
    """Find top loops ranked by NCC and by gap length.

    Returns:
        Tuple of (by_ncc, by_gap) where each is a list of
        (start_frame, end_frame, gap_seconds, ncc) tuples.
    """
    n = len(rows)
    if n < 2:
        return [], []

    ncc_matrix, gaps = compute_ncc_matrix(rows, frame_indices)

    # Upper triangle only (j > i)
    mask = gaps > 0
    ai, bi = np.where(mask)
    nccs = ncc_matrix[ai, bi]
    gap_vals = gaps[ai, bi]

    # Top by NCC
    ncc_order = np.argsort(nccs)[::-1][:top_n]
    by_ncc = [
        (frame_indices[ai[k]], frame_indices[bi[k]], float(gap_vals[k]) / fps, float(nccs[k]))
        for k in ncc_order
    ]

    # Top by gap (longest first), with minimum NCC > 0.5 to filter garbage
    valid = nccs > 0.5
    if valid.any():
        valid_idx = np.where(valid)[0]
        gap_order = np.argsort(gap_vals[valid_idx])[::-1][:top_n]
        by_gap = [
            (frame_indices[ai[valid_idx[k]]], frame_indices[bi[valid_idx[k]]],
             float(gap_vals[valid_idx[k]]) / fps, float(nccs[valid_idx[k]]))
            for k in gap_order
        ]
    else:
        by_gap = []

    return by_ncc, by_gap


def analyze_video(video_path, threshold, downsample):
    """Analyze a single video for loop segments.

    Returns:
        Dict with video info and loop detection results.
    """
    rows, frame_indices, fps, total = extract_frames(video_path, downsample)
    dur = total / fps if fps > 0 else 0

    result = {
        "file": os.path.basename(video_path),
        "total_frames": total,
        "fps": fps,
        "duration": dur,
        "loop": None,
    }

    loop = find_best_loop(rows, frame_indices, threshold)
    if loop:
        start, end, ncc = loop
        result["loop"] = {
            "start_frame": start,
            "end_frame": end,
            "duration": (end - start) / fps,
            "ncc": ncc,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect loop segments in video files")
    parser.add_argument("path", help="Video file or directory of videos")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"NCC threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE,
                        help=f"Extract every Nth frame (default: {DEFAULT_DOWNSAMPLE})")
    parser.add_argument("--detail", action="store_true",
                        help="Show top 10 loops by NCC and by duration (single file only)")
    parser.add_argument("--target-length", type=float, default=None,
                        help="Find best NCC loop closest to this duration in seconds")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        videos = sorted(
            glob.glob(os.path.join(args.path, "*.mp4"))
            + glob.glob(os.path.join(args.path, "*.mov"))
        )
    else:
        videos = [args.path]

    if args.detail and len(videos) == 1:
        path = videos[0]
        print(f"Analyzing: {os.path.basename(path)}")
        t0 = time.time()
        rows, frame_indices, fps, total = extract_frames(path, args.downsample)
        dur = total / fps if fps > 0 else 0
        print(f"Frames: {total}, FPS: {fps:.0f}, Duration: {dur:.1f}s, Sampled: {len(rows)}")

        by_ncc, by_gap = find_top_loops(rows, frame_indices, fps)

        print(f"\nTop 10 by NCC (highest similarity):")
        print(f"  {'Start':>6} {'End':>6} {'Duration':>9} {'NCC':>7}")
        for start, end, gap_s, ncc in by_ncc:
            print(f"  f{start:>5} f{end:>5} {gap_s:>8.1f}s {ncc:>6.1%}")

        print(f"\nTop 10 by duration (longest loops with NCC > 50%):")
        print(f"  {'Start':>6} {'End':>6} {'Duration':>9} {'NCC':>7}")
        for start, end, gap_s, ncc in by_gap:
            print(f"  f{start:>5} f{end:>5} {gap_s:>8.1f}s {ncc:>6.1%}")

        if args.target_length:
            result = find_best_for_target(rows, frame_indices, fps, args.target_length)
            if result:
                print(f"\nBest loop near {args.target_length:.0f}s: f{result[0]}->f{result[1]} ({result[2]:.1f}s) NCC={result[3]:.1%}")
            else:
                print(f"\nNo loop found near {args.target_length:.0f}s")

        # Check if highest NCC pair (>= 1s) is loopable
        best = find_best_for_target(rows, frame_indices, fps, target_seconds=dur)
        if best and best[3] >= 0.97:
            raw_loopable = abs(best[2] - dur) <= 0.5
            print(f"\nLoopable: YES (best NCC={best[3]:.1%}, {best[2]:.1f}s)")
            if raw_loopable:
                print(f"Original clip is loopable as-is (within 0.5s of {dur:.1f}s)")
        else:
            ncc_str = f"{best[3]:.1%}" if best else "N/A"
            print(f"\nLoopable: NO (best NCC={ncc_str})")

        print(f"\nDone in {time.time() - t0:.1f}s")
        return

    print(f"{'File':<55} {'Dur':>5} {'Best Loop':>22} {'NCC':>7} {'Loop?'}")
    print("-" * 100)

    t0 = time.time()
    for path in videos:
        result = analyze_video(path, args.threshold, args.downsample)
        name = result["file"]
        dur = result["duration"]

        if result["loop"]:
            lp = result["loop"]
            loop_str = f"f{lp['start_frame']}->{lp['end_frame']} ({lp['duration']:.1f}s)"
            tag = "YES" if lp["ncc"] >= 0.90 else "~"
            print(f"{name:<55} {dur:>4.1f}s {loop_str:>22} {lp['ncc']:>6.1%}  {tag}")
        else:
            print(f"{name:<55} {dur:>4.1f}s {'none found':>22}         no")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
