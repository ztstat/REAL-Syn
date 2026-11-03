"""Recreate chronological experiment figures and tables from saved results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


BUNDLE_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path,
        help="Folder containing grid_summary.csv; default: latest completed local run.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure/table destination; default: visualizations/ beside analysis/.",
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-style", choices=("stix", "computer-modern"))
    return parser


def resolve_analysis_dir(requested):
    if requested is not None:
        directory = requested.expanduser().resolve()
    else:
        candidates = sorted(
            path.parent
            for path in (BUNDLE_ROOT / "outputs").glob("strict_*/analysis/grid_summary.csv")
            if (path.parent.parent / "analysis_complete.json").is_file()
        )
        if not candidates:
            raise FileNotFoundError(
                "No completed local run found. Run run_experiment.py first, "
                "or provide --analysis-dir containing grid_summary.csv."
            )
        directory = candidates[-1]
    if not (directory / "grid_summary.csv").is_file():
        raise FileNotFoundError(f"Missing {directory / 'grid_summary.csv'}")
    return directory


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "temporal-visualization-mpl"))
    from experiment_core.grid_visualizations import render_heatmaps

    analysis_dir = resolve_analysis_dir(args.analysis_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None else analysis_dir.parent / "visualizations"
    )
    render_heatmaps(analysis_dir, output_dir, args.formats, args.dpi, args.font_style)
    source = Path(os.path.relpath(analysis_dir, output_dir)).as_posix()
    lines = [
        "# Chronological experiment figures and tables", "",
        f"Source analysis: `{source}`.", "",
        "- `all/`: full-test MSE comparisons, covariance, acceptance, eta, best-method maps, and Markdown tables.",
        "- `tails/`: M1.0+ tail-MSE comparisons, best-method maps, and mean ± sample-SD tables.",
    ]
    lines.extend([
        "",
        "Mean ± SD tables recover sample SD from each saved SE and repetition count. "
        "The guarded full-test map uses MSE when the relative difference ± 1 SD excludes zero; "
        "otherwise it compares empirical covariance among the four synthetic methods. "
        "Real-only remains eligible for the MSE comparison.",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Finished: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
