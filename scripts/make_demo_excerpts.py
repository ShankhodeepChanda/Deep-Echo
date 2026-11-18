"""
Cut small demo excerpts of the bundled recordings for hosted deployments.

The full recordings in data/raw (1.4-1.8 GB each) are not in git, so a hosted
copy of the dashboard (e.g. Streamlit Community Cloud) has no observation data.
This script writes, for every recording that has stored held-out rankings, a
region of about ``--budget-mb`` megabytes chosen to hold the most highly
ranked held-out tiles, into data/samples/ (which is committed).

Each excerpt is checked before it is kept: windows must be bit-identical to the
original, and the model must reproduce the stored scores of its top tiles.

    python scripts/make_demo_excerpts.py              # write excerpts
    python scripts/make_demo_excerpts.py --dry-run    # only report the chosen regions
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, RESULTS_DIR, SAMPLES_DATA_DIR
from src.observatory.excerpts import choose_window, write_excerpt
from src.observatory.observations import _safe_id, load_results_index
from src.observatory.provenance import TILE_F, TILE_T, heldout_origins, parse_test_tile_index
from src.observatory.scoring import load_model, score_windows
from src.observatory.sources import FilterbankSource


def ranked_heldout(meta, rankings_path: Path):
    """(origins, scores) of the stored held-out tiles, best first, in original-file coordinates."""
    df = pd.read_csv(rankings_path).dropna(subset=["reconstruction_error"])
    df = df.sort_values("reconstruction_error", ascending=False, kind="stable")
    idx = np.array([parse_test_tile_index(t) for t in df["test_tile_id"]], dtype=np.int64)
    return heldout_origins(meta.n_times, meta.nchans)[idx], df["reconstruction_error"].to_numpy(float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-mb", type=float, default=25.0, help="target data size per excerpt (default 25)")
    ap.add_argument("--out", type=Path, default=SAMPLES_DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = load_results_index(RESULTS_DIR)
    files = sorted(RAW_DATA_DIR.glob("*.fil"))
    if not files:
        print(f"No .fil recordings in {RAW_DATA_DIR}; nothing to excerpt.")
        return 1
    bundle = None if args.dry_run else load_model()
    budget = int(args.budget_mb * 1024 * 1024)
    failures = 0

    for path in files:
        src = FilterbankSource(path)
        m = src.meta
        info = results.get(m.source_name)
        if info is None or info.rankings_path is None:
            print(f"- {path.name}: no stored rankings, skipped")
            continue
        origins, scores = ranked_heldout(m, info.rankings_path)
        t0, nt, f0, nf = choose_window(m.n_times, m.nchans, origins, budget)
        inside = (
            (origins[:, 0] >= t0) & (origins[:, 0] + TILE_T <= t0 + nt)
            & (origins[:, 1] >= f0) & (origins[:, 1] + TILE_F <= f0 + nf)
        )
        ranks = np.flatnonzero(inside) + 1
        f_hi, f_lo = sorted([m.fch1_mhz + f0 * m.foff_mhz, m.fch1_mhz + (f0 + nf - 1) * m.foff_mhz], reverse=True)
        print(
            f"- {m.source_name}: t[{t0}:{t0 + nt}] ({nt * m.tsamp_s:.1f} s of {m.duration_s:.1f} s), "
            f"ch[{f0}:{f0 + nf}] ({f_lo:.3f}-{f_hi:.3f} MHz), {nt * nf * 4 / 1e6:.1f} MB; "
            f"{inside.sum()} of {len(origins)} held-out tiles inside, best ranks {ranks[:8].tolist()}"
        )
        if args.dry_run:
            continue

        out = args.out / f"{_safe_id(m.source_name)}.fil"
        ex = write_excerpt(path, out, t0, nt, f0, nf, extra={"heldout_tiles_inside": int(inside.sum())})
        excerpt = FilterbankSource(out)

        # 1. Data: the top windows are bit-identical to the original.
        local = ex.to_local(origins[inside])
        same = all(
            np.array_equal(src.read_window(int(a), TILE_T, int(b), TILE_F), excerpt.read_window(int(c), TILE_T, int(d), TILE_F))
            for (a, b), (c, d) in zip(origins[inside][:5], local[:5])
        )
        # 2. Model: re-scoring the top tiles in the excerpt reproduces the stored scores.
        fresh = np.asarray(score_windows(bundle, [excerpt.read_window(int(c), TILE_T, int(d), TILE_F) for c, d in local[:5]]))
        diff = float(np.max(np.abs(fresh - scores[inside][:5])))
        ok = same and diff <= 1e-4
        failures += not ok
        print(f"  wrote {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size / 1e6:.1f} MB): "
              f"windows identical={same}, max score difference {diff:.2e} -> {'OK' if ok else 'FAILED'}")
        if not ok:
            out.unlink(missing_ok=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
