"""
Robotic Vision AI — Evaluation Dashboard
Based on outputs/evaluation_report.json + outputs/predictions/*.json

Run: streamlit run eval_dashboard.py --server.port 8502 --server.address 0.0.0.0
"""

import json
import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Robotic Vision AI — Eval",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).parent

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    report = json.load(open(BASE / "outputs/evaluation_report.json"))

    pred_files = sorted(glob.glob(str(BASE / "outputs/predictions/episode_*.json")))
    preds = {}
    for f in pred_files:
        d = json.load(open(f))
        preds[d["video_id"]] = d

    # merge: episodes in report enriched with full prediction features
    rows = []
    for ep in report["per_episode"]:
        vid  = ep["video_id"]
        pred = preds.get(vid, {})
        ef   = pred.get("extracted_features", {})
        rows.append({
            "video_id":         vid,
            "gt_label":         ep["gt_label"],
            "predicted_label":  ep["predicted_label"],
            "correct":          ep["correct"],
            "confidence":       ep["confidence"],
            "gt_reasoning":     ep["gt_reasoning"],
            "reasoning":        pred.get("reasoning", pred.get("failure_reason", "")),
            "final_result":     pred.get("final_result", ep["predicted_label"]),
            "duration_sec":     ef.get("duration_sec", 0),
            "z_rise":           ef.get("z_rise", 0),
            "z_rise_after_grasp": ef.get("z_rise_after_grasp", 0),
            "peak_z":           ef.get("peak_z", 0),
            "z_shape":          ef.get("z_shape", ""),
            "gripper_open_pct": ef.get("gripper_open_pct", 0),
            "gripper_closed_pct": ef.get("gripper_closed_pct", 0),
            "grasp_events":     ef.get("grasp_events", 0),
            "abrupt_count":     ef.get("abrupt_count", 0),
            "motion_quality":   ef.get("motion_quality", ""),
            "total_xy_dist":    ef.get("total_xy_dist", 0),
            "mean_vel":         ef.get("mean_vel", 0),
            "max_vel":          ef.get("max_vel", 0),
        })

    df = pd.DataFrame(rows)
    return report, preds, df

report, preds, df = load_data()
summary = report["summary"]

# ── derived metrics ───────────────────────────────────────────────────────────
# All GT = failure. Binary: failure=positive, success=negative.
TP = int(((df.predicted_label == "failure") & (df.gt_label == "failure")).sum())
FP = int(((df.predicted_label == "failure") & (df.gt_label == "success")).sum())
TN = int(((df.predicted_label == "success") & (df.gt_label == "success")).sum())
FN = int(((df.predicted_label == "success") & (df.gt_label == "failure")).sum())

accuracy  = summary["accuracy"]
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

# ── header ────────────────────────────────────────────────────────────────────
st.title("🦾 Robotic Vision AI — Evaluation Report")
st.caption(
    "Model: **Cosmos-Reason2** · theconstruct-ai/pick_place_cube** · "
    f"Episodes evaluated: **{summary['total_episodes']}** · "
    "Ground truth: Labels in episodes.json"
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI TILES
# ═══════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{accuracy*100:.1f}%",  f"{summary['correct']}/{summary['total_episodes']} correct")
c2.metric("Precision", f"{precision*100:.1f}%", "failure class")
c3.metric("Recall",    f"{recall*100:.1f}%",    "failure class")
c4.metric("F1 Score",  f"{f1*100:.1f}%")
c5.metric("GT Failures", f"{(df.gt_label=='failure').sum()}", "all episodes are failure")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — METRICS BAR + CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════
col_bar, col_cm = st.columns([1, 1])

with col_bar:
    st.subheader("Classification Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score":  [accuracy, precision, recall, f1],
        "Color":  ["#4C9BE8", "#F5A623", "#7ED321", "#E74C3C"],
    })
    fig_bar = go.Figure(go.Bar(
        x=metrics_df["Metric"],
        y=metrics_df["Score"],
        marker_color=metrics_df["Color"],
        text=[f"{v*100:.1f}%" for v in metrics_df["Score"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"),
        template="plotly_dark",
        height=340,
        margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "⚠️ All 15 GT episodes are **failure**. "
        "Precision = 1.0 (both predicted failures were correct). "
        "Recall = 0.13 (model predicted 'success' for 13 actual failures — "
        "confident false positives)."
    )

with col_cm:
    st.subheader("Confusion Matrix")
    cm_data = [[TN, FP], [FN, TP]]
    labels  = ["success", "failure"]
    totals  = [[TN + FP, 0], [FN + TP, 0]]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_data,
        x=["Pred: success", "Pred: failure"],
        y=["GT: success",   "GT: failure"],
        colorscale=[[0, "#1a1a2e"], [0.5, "#16213e"], [1, "#0f3460"]],
        showscale=False,
        text=[[f"TN={TN}", f"FP={FP}"], [f"FN={FN}", f"TP={TP}"]],
        texttemplate="%{text}",
        textfont=dict(size=20, color="white"),
    ))
    fig_cm.add_annotation(x="Pred: success", y="GT: success",
        text="True Negative", showarrow=False, yshift=-18,
        font=dict(size=11, color="#aaa"))
    fig_cm.add_annotation(x="Pred: failure", y="GT: success",
        text="False Positive", showarrow=False, yshift=-18,
        font=dict(size=11, color="#aaa"))
    fig_cm.add_annotation(x="Pred: success", y="GT: failure",
        text="False Negative", showarrow=False, yshift=-18,
        font=dict(size=11, color="#aaa"))
    fig_cm.add_annotation(x="Pred: failure", y="GT: failure",
        text="True Positive", showarrow=False, yshift=-18,
        font=dict(size=11, color="#aaa"))
    fig_cm.update_layout(
        template="plotly_dark", height=340,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTION vs GT TABLE
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Per-Episode Results")

display_df = df[["video_id","gt_label","predicted_label","correct","confidence"]].copy()
display_df["✓"] = display_df["correct"].map({True: "✅", False: "❌"})
display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2f}")
display_df = display_df.rename(columns={
    "video_id": "Episode", "gt_label": "GT", "predicted_label": "Predicted", "confidence": "Conf"
})

def color_row(row):
    bg = "#1a3a1a" if row["✓"] == "✅" else "#3a1a1a"
    return [f"background-color: {bg}"] * len(row)

st.dataframe(
    display_df[["Episode","GT","Predicted","✓","Conf"]],
    use_container_width=True,
    hide_index=True,
    height=380,
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Robot Motion Features — Correct vs Wrong Predictions")

col_z, col_grip = st.columns(2)

with col_z:
    fig_z = go.Figure()
    for correct, label, color in [(True, "Correct (TP)", "#2ECC71"), (False, "Wrong (FN)", "#E74C3C")]:
        sub = df[df["correct"] == correct]
        fig_z.add_trace(go.Box(
            y=sub["z_rise"], name=label,
            marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
        ))
    fig_z.update_layout(
        title="Z-Axis Rise (m) — pre-grasp",
        yaxis_title="z_rise (m)",
        template="plotly_dark", height=320, margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_z, use_container_width=True)

with col_grip:
    fig_grip = go.Figure()
    for correct, label, color in [(True, "Correct (TP)", "#2ECC71"), (False, "Wrong (FN)", "#E74C3C")]:
        sub = df[df["correct"] == correct]
        fig_grip.add_trace(go.Box(
            y=sub["gripper_closed_pct"], name=label,
            marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
        ))
    fig_grip.update_layout(
        title="Gripper Closed % — per episode",
        yaxis_title="% frames gripper closed",
        template="plotly_dark", height=320, margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_grip, use_container_width=True)

# Confidence distribution
col_conf, col_abrupt = st.columns(2)

with col_conf:
    fig_conf = px.histogram(
        df, x="confidence", nbins=8,
        color="correct",
        color_discrete_map={True: "#2ECC71", False: "#E74C3C"},
        title="Confidence Distribution (correct vs wrong)",
        labels={"correct": "Correct", "confidence": "Confidence Score"},
        template="plotly_dark",
        barmode="overlay",
    )
    fig_conf.update_traces(opacity=0.75)
    fig_conf.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig_conf, use_container_width=True)
    st.caption(
        "Most wrong predictions have **high confidence (0.97)** — "
        "the model is confidently wrong. Correct predictions have confidence=0 "
        "(fallback mode), suggesting the model's confidence score is **not calibrated**."
    )

with col_abrupt:
    fig_abrupt = go.Figure()
    for correct, label, color in [(True, "Correct", "#2ECC71"), (False, "Wrong", "#E74C3C")]:
        sub = df[df["correct"] == correct]
        fig_abrupt.add_trace(go.Box(
            y=sub["abrupt_count"], name=label,
            marker_color=color, boxpoints="all", jitter=0.4,
        ))
    fig_abrupt.update_layout(
        title="Abrupt Velocity Changes per Episode",
        yaxis_title="abrupt_count",
        template="plotly_dark", height=300, margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_abrupt, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — SAMPLE JSON INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("📄 Sample Prediction JSON Inspector")

ep_ids = sorted(preds.keys())
selected = st.selectbox("Select episode", ep_ids, index=ep_ids.index("episode_000013.mp4") if "episode_000013.mp4" in ep_ids else 0)

col_json, col_notes = st.columns([3, 2])

with col_json:
    pred_data = preds[selected]
    st.json(pred_data)

with col_notes:
    ef = pred_data.get("extracted_features", {})
    st.markdown("**Key Signals**")

    sig_df = pd.DataFrame({
        "Feature": [
            "Duration (s)", "Z Rise (m)", "Z Rise after Grasp (m)",
            "Peak Z (m)", "Z Shape",
            "Gripper Closed %", "Grasp Events",
            "Abrupt Velocity Changes", "Motion Quality",
        ],
        "Value": [
            f"{ef.get('duration_sec', 0):.1f}",
            f"{ef.get('z_rise', 0):.4f}",
            f"{ef.get('z_rise_after_grasp', 0):.4f}",
            f"{ef.get('peak_z', 0):.4f}",
            ef.get("z_shape", ""),
            f"{ef.get('gripper_closed_pct', 0):.1f}%",
            str(ef.get("grasp_events", 0)),
            str(ef.get("abrupt_count", 0)),
            ef.get("motion_quality", ""),
        ],
    })
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    gt   = pred_data.get("ground_truth_label", "?")
    pred = pred_data.get("final_result", "?")
    ok   = "❌ WRONG" if gt == pred else "✅ CORRECT"
    st.markdown(f"**GT:** `{gt}` &nbsp; **Predicted:** `{pred}` &nbsp; {ok}")

    reasoning = pred_data.get("reasoning") or pred_data.get("failure_reason", "")
    if reasoning:
        st.info(f"**Model Reasoning:** {reasoning[:300]}{'…' if len(reasoning)>300 else ''}")

    st.markdown(f"**GT Reasoning:** {ef.get('gt_reasoning','')}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("📋 Evaluation Summary & Diagnosis")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.error(
        f"**Root Cause of Low Recall ({recall*100:.0f}%):**  \n"
        "All 15 ground-truth episodes are **failure** (robot grasped but did not lift — "
        "z only rose 0.000m after grasp, threshold ≥0.04m).  \n"
        "Cosmos predicted **'success'** for 13 of these with high confidence (~0.97), "
        "because the video *looks like* a successful pick-and-place visually "
        "(arm moves, gripper closes) but the z-sensor shows the cube was never actually lifted."
    )
with col_d2:
    st.success(
        f"**What the model gets right:**  \n"
        f"Precision = {precision*100:.0f}% — both episodes it called 'failure' *were* failures.  \n\n"
        "**Gap:** The model lacks access to the sensor z-channel at inference time — "
        "it sees video frames only. The failure mode (z_rise_after_grasp = 0.0m) is "
        "**not visually distinguishable** from success in the RGB frames alone.  \n\n"
        "**Recommendation:** Feed z-trajectory or depth signal as a numeric context "
        "alongside the video (as done in the robot-task-monitoring pipeline)."
    )

# ── footer ────────────────────────────────────────────────────────────────────
st.caption(
    f"Episodes: {summary['total_episodes']} · "
    f"Correct: {summary['correct']} · "
    f"Accuracy: {accuracy*100:.1f}% · "
    f"Precision: {precision*100:.1f}% · "
    f"Recall: {recall*100:.1f}% · "
    f"F1: {f1*100:.1f}%"
)
