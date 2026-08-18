"""Phase 1 gate: timestamp truth and complete-duration accounting (PRD 6.1, 18).

The first requirement in PRD 18 is that the complete decodable duration is
processed, or every missing range is explicitly failed. That claim is only worth
anything if ingest actually returns every frame, so the central check here
compares the ingest frame count against a plain full decode of the same file,
on real corpus video.

That comparison is not decorative. It caught ingest silently dropping the last
**11 frames of every file** — under 1%, invisible in any summary — because the
decode generator was reconstructed on each iteration and the decoder's internal
buffer was never flushed.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av  # noqa: E402

from pipeline import ingest  # noqa: E402
from pipeline.timestamps import GAP_RATIO, build_index  # noqa: E402

CORPUS = "C:/DrishtiAI"

# Measured across the corpus. The two populations differ by a factor of 150 and
# nothing sits between them, so these are assertions about the data rather than
# about a tuned threshold.
EXPECTED_VFR = {
    "01.Candidate": False,
    "02.Candidate": False,
    "03.CCTV": True,
    "04.CCTV": True,      # 144 frames held 160 ms against a 120 ms majority
    "05.Crowd": True,
    "Seat No. 12": True,
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def plain_decode_count(path: str) -> int:
    """Frame count from the simplest possible full decode, as ground truth."""
    n = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for _ in container.decode(stream):
            n += 1
    return n


def main() -> int:
    ok = True

    print("timestamp sequence analysis")
    steady = build_index([(i * 40.0, i % 25 == 0, "P") for i in range(200)])
    ok &= check("constant pacing is not variable rate", not steady.is_vfr,
                f"cv {steady.interval_cv:.6f}")
    ok &= check("measured fps from PTS", abs(steady.measured_fps - 25.0) < 0.01,
                f"{steady.measured_fps:.3f}")
    ok &= check("duration includes the last frame's own interval",
                abs(steady.duration_ms - 8000.0) < 1e-6, f"{steady.duration_ms}")
    ok &= check("keyframes counted", steady.keyframe_count == 8,
                f"{steady.keyframe_count}")

    mixed = build_index([(i * 40.0 if i < 100 else 4000.0 + (i - 100) * 80.0,
                          False, "P") for i in range(200)])
    ok &= check("mixed pacing is variable rate", mixed.is_vfr,
                f"cv {mixed.interval_cv:.4f}")

    repeated = build_index([(0.0, True, "I"), (40.0, False, "P"),
                            (40.0, False, "P"), (80.0, False, "P")])
    ok &= check("a repeated timestamp is flagged", repeated.repeated_pts_count == 1)

    backwards = build_index([(0.0, True, "I"), (80.0, False, "P"),
                             (40.0, False, "P")])
    ok &= check("a backwards timestamp is flagged",
                backwards.backwards_pts_count == 1)

    print("\ngap detection")
    with_gap = build_index(
        [(i * 40.0, False, "P") for i in range(50)]
        + [(2000.0 + 3000.0 + i * 40.0, False, "P") for i in range(50)])
    ok &= check("one gap found", len(with_gap.gaps) == 1, f"{len(with_gap.gaps)}")
    if with_gap.gaps:
        ok &= check("the gap is roughly the removed span",
                    abs(with_gap.gaps[0].duration_ms - 3000.0) < 80.0,
                    f"{with_gap.gaps[0].duration_ms:.1f} ms")
    ok &= check("decodable duration excludes the gap",
                with_gap.decodable_ms < with_gap.duration_ms,
                f"{with_gap.decodable_ms:.0f} of {with_gap.duration_ms:.0f} ms")
    ok &= check("an interval inside the gap is reported as inside it",
                with_gap.inside_gap(3500.0) is not None)
    ok &= check("an event spanning the gap can be told so",
                len(with_gap.spans_gap(1900.0, 5200.0)) == 1,
                "PRD 8 forbids bridging an event across a decode gap")
    ok &= check("gap intervals do not inflate the variable-rate verdict",
                not with_gap.is_vfr,
                f"cv {with_gap.interval_cv:.6f}; one dropped span is not VFR")

    # A gap must be far enough outside normal pacing that VFR jitter never
    # reaches it. Below the ratio it is pacing; above it, frames are missing.
    just_under = build_index(
        [(i * 40.0, False, "P") for i in range(20)]
        + [(760.0 + 40.0 * (GAP_RATIO - 1), False, "P")])
    ok &= check("an interval just under the ratio is not a gap",
                not just_under.gaps, f"{len(just_under.gaps)}")

    print("\nartifact provenance")
    index = build_index([(i * 40.0, False, "P") for i in range(100)])
    nearest = index.nearest(1234.0)
    ok &= check("any timestamp maps back to a source frame",
                nearest is not None and abs(nearest.pts_ms - 1240.0) < 1e-6,
                f"{nearest.pts_ms if nearest else None}")
    ok &= check("an empty index claims no frame",
                build_index([]).nearest(0.0) is None,
                "an artifact that cannot name its source frame must not claim one")

    print("\ncomplete-duration accounting on real footage (PRD 18)")
    files = sorted(glob.glob(f"{CORPUS}/*.mkv") + glob.glob(f"{CORPUS}/*.mp4"))
    if not files:
        print("  SKIP: corpus not available")
    else:
        for path in files:
            name = Path(path).name
            result = ingest.decode(path)
            expected = plain_decode_count(path)
            # The regression that matters: every frame, not almost every frame.
            ok &= check(f"{name[:28]:30} every frame decoded",
                        result.index.count == expected,
                        f"{result.index.count} vs {expected}")
            ok &= check(f"{name[:28]:30} no spurious decode errors",
                        not result.index.decode_errors,
                        str(result.index.decode_errors[:1]))
            ok &= check(f"{name[:28]:30} per-frame stats for every frame",
                        len(result.stats) == result.index.count,
                        f"{len(result.stats)} vs {result.index.count}")

            key = next((k for k in EXPECTED_VFR if name.startswith(k)), None)
            if key is not None:
                ok &= check(f"{name[:28]:30} rate verdict",
                            result.index.is_vfr == EXPECTED_VFR[key],
                            f"vfr={result.index.is_vfr} "
                            f"cv={result.index.interval_cv:.4f}")

        print("\n  declared frame rates are checked, not trusted")
        untrustworthy = []
        for path in files:
            result = ingest.decode(path)
            if not result.fps_declaration_trustworthy:
                declared, measured = result.declared_vs_measured_fps
                untrustworthy.append(f"{Path(path).name[:20]} "
                                     f"declares {declared:.2f}, measures {measured:.2f}")
        ok &= check("at least one file declares a rate it does not keep",
                    len(untrustworthy) >= 1, "; ".join(untrustworthy))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
