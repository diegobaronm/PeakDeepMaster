#!/usr/bin/env python3
"""Collect result plots into a paper-ready directory structure.

Edit PLOT_MAP below: keys are source paths relative to results/,
values are destination paths relative to the paper output directory.
Directories are created on the fly as each file is copied.
"""
import shutil
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "paper_plots"

# ──────────────────────────────────────────────────────────────────────
#  PLOT_MAP: source (under results/) → destination (under paper_plots/)
#
#  Organise by paper section / figure topic.  Edit freely.
# ──────────────────────────────────────────────────────────────────────
PLOT_MAP: dict[str, str] = {

    # ── Input distributions: 1D coupling ──────────────────────────────

    # ── Input distributions: 1D mass ──────────────────────────────────
    "input_plots/1D_mass/EVENT_GLOBAL_mttbar.pdf":                             "ForPaper/EVENT_GLOBAL_mttbar.pdf",

    # ── Performance: 1D coupling  ──────────────
    "performance/1D_coupling_PP_256_correct_split/test/roc_test_by_category.pdf":       "ForPaper/1D_coupling_PP_performance.pdf",
    "performance/1D_coupling_PN_256_correct_split/test/roc_test_by_category.pdf":    "ForPaper/1D_coupling_PN_performance.pdf",

    # ── Performance: 1D mass  ─────────────────────────────────────
    "performance/1D_mass_PP_256_correct_split/test/roc_test_by_category.pdf":           "ForPaper/1D_mass_PP_performance.pdf",
    "performance/1D_mass_PN_256_correct_split/test/roc_test_by_category.pdf":        "ForPaper/1D_mass_PN_performance.pdf",

    # ── Inference: 1D coupling (zoom = fine-grained scan around truth)
    # 0.5
    "inference/1D_coupling_0p5_256_correct_split/chi2_scan.pdf":                 "ForPaper/1D_coupling_0p5/chi2_scan.pdf",
    "inference/1D_coupling_0p5_256_correct_split/hypothesis_shape.pdf":          "ForPaper/1D_coupling_0p5/hypothesis_shape.pdf",
    "inference/1D_coupling_0p5_256_zoom_correct_split/inference_review.pdf":          "ForPaper/1D_coupling_0p5/inference_review.pdf",
    "inference/1D_coupling_0p5_256_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/1D_coupling_0p5/pseudo_experiment_coupling.pdf",
    # 1.1
    "inference/1D_coupling_1p1_256_correct_split/chi2_scan.pdf":                 "ForPaper/1D_coupling_1p1/chi2_scan.pdf",
    "inference/1D_coupling_1p1_256_correct_split/hypothesis_shape.pdf":          "ForPaper/1D_coupling_1p1/hypothesis_shape.pdf",
    "inference/1D_coupling_1p1_256_zoom_correct_split/inference_review.pdf":          "ForPaper/1D_coupling_1p1/inference_review.pdf",
    "inference/1D_coupling_1p1_256_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/1D_coupling_1p1/pseudo_experiment_coupling.pdf",

    # ── Inference: 1D mass (zoom = fine-grained scan around truth)
    # 650
    "inference/mass_650_256_correct_split/chi2_scan.pdf":                 "ForPaper/1D_mass_650/chi2_scan.pdf",
    "inference/mass_650_256_correct_split/hypothesis_shape.pdf":          "ForPaper/1D_mass_650/hypothesis_shape.pdf",
    "inference/mass_650_256_zoom_correct_split/inference_review.pdf":          "ForPaper/1D_mass_650/inference_review.pdf",
    "inference/mass_650_256_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/1D_mass_650/pseudo_experiment_mass.pdf",
    # 870
    "inference/mass_870_256_correct_split/chi2_scan.pdf":                 "ForPaper/1D_mass_870/chi2_scan.pdf",
    "inference/mass_870_256_correct_split/hypothesis_shape.pdf":          "ForPaper/1D_mass_870/hypothesis_shape.pdf",
    "inference/mass_870_256_zoom_correct_split/inference_review.pdf":          "ForPaper/1D_mass_870/inference_review.pdf",
    "inference/mass_870_256_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/1D_mass_870/pseudo_experiment_mass.pdf",


    # ── Inference: 2D coupling×mass  ────────────────────────────
    # 0.5 - 650
    "inference/coupling_0p5_hmass_650_correct_split/chi2_scan_heatmap.pdf":                 "ForPaper/2D_0p5_650/chi2_scan_heatmap.pdf",
    "inference/coupling_0p5_hmass_650_correct_split/hypothesis_shape.pdf":          "ForPaper/2D_0p5_650/hypothesis_shape.pdf",
    "inference/coupling_0p5_hmass_650_zoom_correct_split/inference_review.pdf":          "ForPaper/2D_0p5_650/inference_review.pdf",
    "inference/coupling_0p5_hmass_650_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/2D_0p5_650/pseudo_experiment_coupling.pdf",
    "inference/coupling_0p5_hmass_650_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/2D_0p5_650/pseudo_experiment_mass.pdf",
    # 0.5 - 870
    "inference/coupling_0p5_hmass_870_correct_split/chi2_scan_heatmap.pdf":                 "ForPaper/2D_0p5_870/chi2_scan_heatmap.pdf",
    "inference/coupling_0p5_hmass_870_correct_split/hypothesis_shape.pdf":          "ForPaper/2D_0p5_870/hypothesis_shape.pdf",
    "inference/coupling_0p5_hmass_870_zoom_correct_split/inference_review.pdf":          "ForPaper/2D_0p5_870/inference_review.pdf",
    "inference/coupling_0p5_hmass_870_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/2D_0p5_870/pseudo_experiment_coupling.pdf",
    "inference/coupling_0p5_hmass_870_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/2D_0p5_870/pseudo_experiment_mass.pdf",
    # 1.1 - 650
    "inference/coupling_1p1_hmass_650_correct_split/chi2_scan_heatmap.pdf":                 "ForPaper/2D_1p1_650/chi2_scan_heatmap.pdf",
    "inference/coupling_1p1_hmass_650_correct_split/hypothesis_shape.pdf":          "ForPaper/2D_1p1_650/hypothesis_shape.pdf",
    "inference/coupling_1p1_hmass_650_zoom_correct_split/inference_review.pdf":          "ForPaper/2D_1p1_650/inference_review.pdf",
    "inference/coupling_1p1_hmass_650_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/2D_1p1_650/pseudo_experiment_coupling.pdf",
    "inference/coupling_1p1_hmass_650_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/2D_1p1_650/pseudo_experiment_mass.pdf",
    # 1.1 - 870
    "inference/coupling_1p1_hmass_870_correct_split/chi2_scan_heatmap.pdf":                 "ForPaper/2D_1p1_870/chi2_scan_heatmap.pdf",
    "inference/coupling_1p1_hmass_870_correct_split/hypothesis_shape.pdf":          "ForPaper/2D_1p1_870/hypothesis_shape.pdf",
    "inference/coupling_1p1_hmass_870_zoom_correct_split/inference_review.pdf":          "ForPaper/2D_1p1_870/inference_review.pdf",
    "inference/coupling_1p1_hmass_870_zoom_correct_split/pseudo_experiment_$C_e$.pdf": "ForPaper/2D_1p1_870/pseudo_experiment_coupling.pdf",
    "inference/coupling_1p1_hmass_870_zoom_correct_split/pseudo_experiment_$m(S)$.pdf": "ForPaper/2D_1p1_870/pseudo_experiment_mass.pdf",

    # ── Inference: width robustness studies (holdout width ≠ training)
    # Natural width
    "inference/coupling_0p7_hmass_950_width_2_correct_split/chi2_scan_heatmap.pdf":                 "ForPaper/Width_Tests/FromModel/chi2_scan_heatmap.pdf",
    "inference/coupling_0p7_hmass_950_width_2_correct_split/hypothesis_shape.pdf":          "ForPaper/Width_Tests/FromModel/hypothesis_shape.pdf",
    "inference/coupling_0p7_hmass_950_width_2_correct_split/inference_review.pdf":          "ForPaper/Width_Tests/FromModel/inference_review.pdf",
    # W5
    "inference/coupling_0p7_hmass_950_width_5_correct_split/hypothesis_shape.pdf":          "ForPaper/Width_Tests/W5/hypothesis_shape.pdf",
    "inference/coupling_0p7_hmass_950_width_5_correct_split/inference_review.pdf":          "ForPaper/Width_Tests/W5/inference_review.pdf",
    # W10
    "inference/coupling_0p7_hmass_950_width_10_correct_split/hypothesis_shape.pdf":          "ForPaper/Width_Tests/W10/hypothesis_shape.pdf",
    "inference/coupling_0p7_hmass_950_width_10_correct_split/inference_review.pdf":          "ForPaper/Width_Tests/W10/inference_review.pdf",
    # W15
    "inference/coupling_0p7_hmass_950_width_15_correct_split/hypothesis_shape.pdf":          "ForPaper/Width_Tests/W15/hypothesis_shape.pdf",
    "inference/coupling_0p7_hmass_950_width_15_correct_split/inference_review.pdf":          "ForPaper/Width_Tests/W15/inference_review.pdf",
    # W30
    "inference/coupling_0p7_hmass_950_width_30_correct_split/hypothesis_shape.pdf":          "ForPaper/Width_Tests/W30/hypothesis_shape.pdf",
    "inference/coupling_0p7_hmass_950_width_30_correct_split/inference_review.pdf":          "ForPaper/Width_Tests/W30/inference_review.pdf",


    # ── Widths vs couplings standalone figure ─────────────────────────
    "widths_vs_couplings.pdf":                                                 "ForPaper/widths_vs_couplings.pdf",
}


def main() -> None:
    output_dir = OUTPUT_DIR
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])

    copied, missing = 0, 0
    for source_rel, dest_rel in PLOT_MAP.items():
        src = RESULTS_DIR / source_rel
        dst = output_dir / dest_rel

        if not src.exists():
            print(f"  MISSING  {source_rel}")
            missing += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"  COPIED   {source_rel}  →  {dest_rel}")

    print(f"\nDone: {copied} copied, {missing} missing out of {len(PLOT_MAP)} entries.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
