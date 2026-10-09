# loopsmith

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/prattsolutions) [![Ko-fi](https://img.shields.io/badge/Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/johnpratt)

Find the longest seamlessly-loopable segment in a video.

Given a clip, `loopsmith` figures out the longest stretch you can pull out and
play on repeat without a visible jump — the point where some later frame looks
almost exactly like an earlier one, so playback can snap back to the start with
no seam. Handy for turning B-roll, animations, or background footage into clean
loops (looping wallpapers, GIFs, video backdrops, and so on).

## How it works

The core of it is a single matrix multiply.

1. Sample every Nth frame, shrink each to a 160px thumbnail, convert to
   grayscale, and z-score normalize it (subtract the mean, divide by the
   standard deviation).
2. Stack the normalized frames into a matrix and multiply it by its own
   transpose. For z-scored vectors, each dot product (divided by the pixel
   count) *is* the normalized cross-correlation between two frames — a
   similarity score from -1 to 1 where 1 means "visually identical." So one
   `mat @ mat.T` produces the score for every pair of frames at once.
3. For each pair of frames, the gap between them is a candidate loop length.
   Among all pairs whose similarity clears a threshold, the one with the
   largest gap wins: those two near-identical frames are the in- and out-points
   of the longest seam-free loop.

Because the expensive step is one BLAS matmul, it stays quick even though it's
effectively comparing every frame against every other frame.

## Install

```bash
pip install loopsmith
```

From source, for development (includes the test deps):

```bash
pip install -e ".[dev]"
```

Needs Python 3.9+, OpenCV, and NumPy — the last two are pulled in automatically.

## Usage (CLI)

```bash
# Analyze a single video
loopsmith clip.mp4

# Batch every .mp4/.mov in a directory
loopsmith ./footage/

# Loosen or tighten the similarity threshold (default 0.85)
loopsmith clip.mp4 --threshold 0.80

# Sample every 5th frame instead of every 3rd (faster, coarser)
loopsmith clip.mp4 --downsample 5

# Detailed report: top 10 loops by similarity and by length
loopsmith clip.mp4 --detail

# Find the best loop closest to a target length, in seconds
loopsmith clip.mp4 --detail --target-length 6
```

Batch mode prints the best loop per file as a table; `--detail` (single file)
prints the top candidates plus a yes/no "is this clip loopable" verdict.

## Usage (library)

```python
from loopsmith import extract_frames, find_best_loop

rows, frame_indices, fps, total = extract_frames("clip.mp4", downsample=3)
loop = find_best_loop(rows, frame_indices, threshold=0.85)
if loop:
    start_frame, end_frame, ncc = loop
    seconds = (end_frame - start_frame) / fps
    print(f"Loop {start_frame}->{end_frame} ({seconds:.1f}s) at {ncc:.1%} similarity")
```

Other helpers: `find_best_for_target()` (the loop closest to a target
duration), `find_top_loops()` (ranked candidates), and `analyze_video()` (a
one-call summary dict).

## Notes & limitations

- **Visual only.** It compares frames, not audio. A visually seamless loop can
  still have an audible seam — check the sound separately if that matters.
- **Cut resolution.** In/out points are accurate to within `--downsample`
  frames, since that's the sampling step. Lower it for tighter cuts at the cost
  of speed.
- **Scales as O(N²) in sampled frames.** It builds an N×N similarity matrix, so
  for long videos raise `--downsample` to keep time and memory in check.
- **It reports, it doesn't cut.** Output is frame indices / timestamps and
  similarity scores; trimming the actual clip (e.g. with ffmpeg) is up to you.

## License

MIT — see [LICENSE](LICENSE).
