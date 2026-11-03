"""Advanced / About the model: diagnostics and developer tools, kept out of the investigation workflow."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RESULTS_DIR
from src.observatory.observations import load_results_index
from ui.components import page_head


def _model_diagnostics() -> None:
    summary_path = RESULTS_DIR / "cnn_autoencoder_summary.json"
    if not summary_path.exists():
        st.info("No training summary is saved. Run the training script to create one.")
        return
    from ui.dev_lab import plot_loss_curve
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training tiles", f"{summary['n_train']:,}" if "n_train" in summary else "n/a")
    c2.metric("Validation tiles", f"{summary['n_val']:,}" if "n_val" in summary else "n/a")
    c3.metric("Final loss (MSE)", f"{summary['final_loss']:.6f}" if "final_loss" in summary else "n/a")
    c4.metric("Calibrated threshold", f"{summary['calibrated_threshold']:.6f}" if "calibrated_threshold" in summary else "n/a")
    if summary.get("loss_history"):
        st.plotly_chart(plot_loss_curve(summary["loss_history"]), width="stretch")

    seq_dir = ARTIFACTS_DIR / "checkpoints" / "sequential_training"
    manifest, stages = seq_dir / "experiment_manifest.json", seq_dir / "sequential_stages_summary.csv"
    if manifest.exists() and stages.exists():
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                seq = json.load(fh)
            st.markdown("**Sequential training**")
            s1, s2 = st.columns(2)
            s1.metric("Recalibrated threshold", f"{seq['final_recalibrated_threshold']:.6f}" if "final_recalibrated_threshold" in seq else "n/a")
            s2.metric("Replay ratio", f"{seq['replay_ratio'] * 100:.0f}%" if "replay_ratio" in seq else "n/a")
            df = pd.read_csv(stages)
            cols = ["stage", "dataset", "telescope", "band", "train_samples", "val_samples", "test_samples", "final_train_loss", "val_mean_error", "stage_threshold"]
            st.dataframe(df[[c for c in cols if c in df.columns]], width="stretch", hide_index=True)
            i1, i2 = st.columns(2)
            for col, name, cap in ((i1, "sequential_loss_curves.png", "Loss during sequential training"), (i2, "sequential_retention_matrix.png", "Retention across datasets")):
                if (FIGURES_DIR / name).exists():
                    col.image(str(FIGURES_DIR / name), caption=cap, width="stretch")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            st.warning(f"Sequential training details could not be read: {exc}")


def _thresholds() -> None:
    index = load_results_index()
    if not index:
        st.info("No stored evaluation results were found.")
        return
    rows = [{
        "Target": k, "Telescope": v.summary.get("telescope"), "Band": v.summary.get("band"),
        "Validated threshold": v.threshold, "Held-out tiles": v.summary.get("n_test_tiles"),
        "Above threshold": v.summary.get("anomalies_detected"), "Rate (%)": v.summary.get("anomaly_rate_pct"),
        "Mean score": v.summary.get("mean_error"), "Max score": v.summary.get("max_error"),
    } for k, v in index.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Validated threshold": st.column_config.NumberColumn(format="%.6f"),
                                "Rate (%)": st.column_config.NumberColumn(format="%.2f"),
                                "Mean score": st.column_config.NumberColumn(format="%.6f"),
                                "Max score": st.column_config.NumberColumn(format="%.6f")})
    st.caption("Each threshold is the 99th percentile of validation scores for that observation. Detection sensitivity in the sidebar is defined relative to it.")
    inj = FIGURES_DIR / "injection_recovery_curve.png"
    if inj.exists():
        st.markdown("**Injection and recovery benchmark**")
        st.image(str(inj), width="stretch")
        st.caption("Recovery rate of synthetic drifting carriers injected at different signal-to-noise ratios.")


def render_developer_tools() -> None:
    st.divider()
    page_head("Advanced / About the model", "Diagnostics and development tools. The investigation pages do not depend on this section.")
    t1, t2, t3 = st.tabs(["Model diagnostics", "Thresholds", "Synthetic training lab"])
    with t1:
        _model_diagnostics()
    with t2:
        _thresholds()
    with t3:
        from ui.dev_lab import render_synthetic_lab
        render_synthetic_lab()
