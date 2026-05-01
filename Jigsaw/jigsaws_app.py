"""
JIGSAWS Knot_Tying — Evaluation Dashboard
==========================================
Streamlit UI for analysing surgical skill assessment results.

Run:
    streamlit run jigsaws_app.py --server.port 8503 --server.address 0.0.0.0

Requires:  jigsaws_pipeline.py  (run --evaluate first to generate outputs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JIGSAWS Surgical Skill Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT       = Path(__file__).parent
JIGSAW_DIR = ROOT / "Jigsaw" / "Knot_Tying"
KIN_DIR    = JIGSAW_DIR / "kinematics" / "AllGestures"
TRANS_DIR  = JIGSAW_DIR / "transcriptions"
META_FILE  = JIGSAW_DIR / "meta_file_Knot_Tying.txt"
OUT_DIR    = ROOT / "jigsaws_outputs"

GESTURE_NAMES = {
    "G1":  "Reach Suture",
    "G11": "Release & Return",
    "G12": "Reach Needle",
    "G13": "Position Needle",
    "G14": "Push Needle",
    "G15": "Pull Suture",
}

GESTURE_COLORS = {
    "G1":  "#4CAF50",
    "G11": "#9E9E9E",
    "G12": "#2196F3",
    "G13": "#FF9800",
    "G14": "#E91E63",
    "G15": "#9C27B0",
}

SKILL_COLORS = {"E": "#2ecc71", "I": "#f39c12", "N": "#e74c3c"}

GRS_COMPONENTS = [
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_meta() -> pd.DataFrame:
    rows = []
    with open(META_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                rows.append({
                    "trial_id":    parts[1],
                    "skill_level": parts[2],
                    "grs_total":   int(parts[3]),
                    **{GRS_COMPONENTS[i]: int(parts[4 + i]) for i in range(6)},
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


@st.cache_data
def load_kinematics(trial_id: str) -> np.ndarray:
    path = KIN_DIR / f"{trial_id}.txt"
    rows = []
    with open(path) as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                rows.append([float(v) for v in vals])
    return np.array(rows, dtype=np.float32)


@st.cache_data
def load_transcription(trial_id: str) -> list[dict]:
    path = TRANS_DIR / f"{trial_id}.txt"
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                segs.append({
                    "start": int(parts[0]),
                    "end":   int(parts[1]),
                    "gesture_id": parts[2],
                    "gesture_name": GESTURE_NAMES.get(parts[2], parts[2]),
                })
    return segs


@st.cache_data
def load_result(trial_id: str) -> dict | None:
    p = OUT_DIR / f"{trial_id}.json"
    if p.exists():
        return json.load(open(p))
    return None


@st.cache_data
def load_summary() -> dict | None:
    p = OUT_DIR / "evaluation_summary.json"
    if p.exists():
        return json.load(open(p))
    return None


def get_available_results() -> list[str]:
    return sorted([p.stem for p in OUT_DIR.glob("Knot_Tying_*.json")])


# ── Plot helpers ──────────────────────────────────────────────────────────────
def kinematic_signals_plot(kin: np.ndarray, trial_id: str) -> go.Figure:
    """4-panel kinematic overview: velocity + gripper for both arms."""
    T  = len(kin)
    t  = np.arange(T) / 30.0   # seconds

    ml_vel = np.linalg.norm(kin[:, 12:15], axis=1)
    mr_vel = np.linalg.norm(kin[:, 31:34], axis=1)
    sl_vel = np.linalg.norm(kin[:, 50:53], axis=1)
    sr_vel = np.linalg.norm(kin[:, 69:72], axis=1)
    ml_g   = kin[:, 18]
    mr_g   = kin[:, 37]
    sl_g   = kin[:, 56]
    sr_g   = kin[:, 75]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Master Left velocity (m/s)",
            "Master Right velocity (m/s)",
            "Master Gripper angles (rad)",
            "Slave velocity (m/s)",
        ],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(x=t, y=ml_vel, line=dict(color="#2196F3", width=1),
                             name="ML velocity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=mr_vel, line=dict(color="#E91E63", width=1),
                             name="MR velocity"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=ml_g, line=dict(color="#2196F3", width=1.5),
                             name="ML gripper"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=mr_g, line=dict(color="#E91E63", width=1.5),
                             name="MR gripper"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=sl_vel, line=dict(color="#4CAF50", width=1),
                             name="SL velocity"), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=sr_vel, line=dict(color="#9C27B0", width=1),
                             name="SR velocity"), row=4, col=1)

    fig.update_layout(
        title=f"Kinematic Signals — {trial_id}",
        height=520, showlegend=True,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    fig.update_xaxes(title_text="Time (s)", row=4, col=1, gridcolor="#333")
    for r in range(1, 5):
        fig.update_yaxes(gridcolor="#333", row=r, col=1)
    return fig


def gesture_timeline_plot(
    gt_segs: list[dict],
    pred_segs: list[dict],
    total_frames: int,
    trial_id: str,
) -> go.Figure:
    """Gantt-style gesture timeline: GT vs predicted."""
    fig = go.Figure()
    T_s = total_frames / 30.0

    def add_bar(segs, y_offset, label_prefix):
        for seg in segs:
            gid   = seg["gesture_id"]
            color = GESTURE_COLORS.get(gid, "#607D8B")
            s     = seg["start"] / 30.0
            e     = seg["end"] / 30.0
            fig.add_shape(
                type="rect",
                x0=s, x1=e, y0=y_offset - 0.4, y1=y_offset + 0.4,
                fillcolor=color, opacity=0.8,
                line=dict(width=0),
            )
            if (e - s) > 0.3:
                fig.add_annotation(
                    x=(s + e) / 2, y=y_offset,
                    text=gid, showarrow=False,
                    font=dict(size=9, color="white"),
                )

    add_bar(gt_segs,   1.0, "GT")
    add_bar(pred_segs, 0.0, "Pred")

    fig.update_layout(
        title=f"Gesture Timeline — {trial_id}",
        xaxis=dict(title="Time (s)", range=[0, T_s], gridcolor="#333"),
        yaxis=dict(
            tickvals=[0, 1], ticktext=["Predicted", "Ground Truth"],
            range=[-0.7, 1.7], gridcolor="#333",
        ),
        height=220,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        showlegend=False,
    )

    # Legend patches
    for gid, name in GESTURE_NAMES.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=GESTURE_COLORS.get(gid, "#607D8B"), symbol="square"),
            name=f"{gid}: {name}",
        ))
    return fig


def grs_radar_plot(
    components_gt: dict,
    components_pred: dict,
    trial_id: str,
) -> go.Figure:
    labels = [c.replace("_", " ").title() for c in GRS_COMPONENTS]
    gt_vals  = [components_gt.get(c, 0) for c in GRS_COMPONENTS]
    pred_vals = [components_pred.get(c, 0) for c in GRS_COMPONENTS]
    labels  += [labels[0]]
    gt_vals  += [gt_vals[0]]
    pred_vals += [pred_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=gt_vals, theta=labels, fill="toself",
                                  name="GT", line_color="#2ecc71", opacity=0.7))
    fig.add_trace(go.Scatterpolar(r=pred_vals, theta=labels, fill="toself",
                                  name="Predicted", line_color="#3498db", opacity=0.5))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
            bgcolor="#0e1117",
        ),
        title=f"GRS Radar — {trial_id}",
        showlegend=True,
        height=360,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def skill_distribution_plot(meta_df: pd.DataFrame) -> go.Figure:
    counts = meta_df["skill_level"].value_counts().reindex(["E", "I", "N"], fill_value=0)
    colors = [SKILL_COLORS[s] for s in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors, text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        title="Skill Level Distribution (full dataset)",
        xaxis=dict(title="Skill Level", gridcolor="#333"),
        yaxis=dict(title="Count", gridcolor="#333"),
        height=280,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def grs_boxplot(meta_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for skill, color in SKILL_COLORS.items():
        vals = meta_df[meta_df["skill_level"] == skill]["grs_total"].tolist()
        fig.add_trace(go.Box(y=vals, name=skill, marker_color=color, boxmean=True))
    fig.update_layout(
        title="GRS Score Distribution by Skill Level",
        yaxis=dict(title="GRS Total", gridcolor="#333"),
        height=300,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def quality_scatter_plot(summary: dict) -> go.Figure:
    rows = summary.get("per_trial", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df, x="grs_gt", y="grs_pred",
        color="skill_gt",
        color_discrete_map=SKILL_COLORS,
        hover_data=["trial_id", "gesture_accuracy", "smoothness"],
        labels={"grs_gt": "GT GRS Total", "grs_pred": "Predicted GRS Total",
                "skill_gt": "Skill Level"},
        title="Predicted vs Ground-Truth GRS Score",
        size_max=12,
    )
    max_grs = max(df["grs_gt"].max(), df["grs_pred"].max(), 6)
    fig.add_shape(type="line", x0=6, y0=6, x1=max_grs, y1=max_grs,
                  line=dict(dash="dash", color="white", width=1))
    fig.update_layout(
        height=380,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def kinematic_quality_heatmap(summary: dict) -> go.Figure:
    rows = summary.get("per_trial", [])
    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("grs_gt")
    metrics = ["smoothness", "bilateral_coordination", "gesture_accuracy"]
    labels  = ["Smoothness", "Bilateral Coord.", "Gesture Accuracy"]

    z  = df[metrics].values.T
    fig = go.Figure(go.Heatmap(
        z=z, x=df["trial_id"].tolist(),
        y=labels, colorscale="RdYlGn", zmin=0, zmax=1,
        text=np.round(z, 3),
        texttemplate="%{text}",
        hovertemplate="Trial: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Quality Metrics Heatmap (sorted by GT GRS)",
        height=280,
        xaxis=dict(tickangle=-30, gridcolor="#333"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def gesture_gantt_all(summary: dict) -> go.Figure:
    """Per-trial gesture composition stacked bar."""
    rows = summary.get("per_trial", [])
    if not rows:
        return go.Figure()

    trial_ids = [r["trial_id"] for r in rows]
    gids = list(GESTURE_NAMES.keys())

    # Build gesture duration table from result files
    data = {gid: [] for gid in gids}
    for r in rows:
        result = load_result(r["trial_id"])
        if not result:
            for gid in gids:
                data[gid].append(0)
            continue
        gt_segs = result.get("gt_gesture_segments", [])
        total   = result["quality"]["task_duration_frames"]
        durations = {gid: 0 for gid in gids}
        for seg in gt_segs:
            gid = seg["gesture_id"]
            if gid in durations:
                durations[gid] += seg["end"] - seg["start"] + 1
        for gid in gids:
            data[gid].append(round(100 * durations[gid] / max(total, 1), 1))

    fig = go.Figure()
    for gid in gids:
        fig.add_trace(go.Bar(
            name=f"{gid}: {GESTURE_NAMES[gid]}",
            x=trial_ids,
            y=data[gid],
            marker_color=GESTURE_COLORS.get(gid, "#607D8B"),
        ))

    fig.update_layout(
        barmode="stack",
        title="Gesture Composition per Trial (% of frames)",
        xaxis=dict(tickangle=-30, gridcolor="#333"),
        yaxis=dict(title="% of total frames", gridcolor="#333"),
        height=360,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", y=-0.35),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(meta_df: pd.DataFrame) -> str:
    st.sidebar.title("🔬 JIGSAWS Dashboard")
    st.sidebar.markdown("**Knot Tying Surgical Skill Assessment**")
    st.sidebar.divider()

    avail = get_available_results()
    if not avail:
        st.sidebar.warning(
            "No results found. Run:\n\n"
            "```\npython jigsaws_pipeline.py --evaluate\n```"
        )
        return ""

    selected = st.sidebar.selectbox(
        "Select Trial", avail, index=0,
        help="Choose a processed trial to analyse"
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Dataset Overview**")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Trials",   len(meta_df))
    col2.metric("Processed", len(avail))

    skill_counts = meta_df["skill_level"].value_counts()
    for skill, color in SKILL_COLORS.items():
        st.sidebar.markdown(
            f"<span style='color:{color}'>■</span> **{skill}**: {skill_counts.get(skill, 0)} trials",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Fusion: Kinematic features + DINOv2 + "
        "Rule-based GRU + Cosmos-Reason2-2B"
    )
    return selected


# ── Tab: Trial Analysis ───────────────────────────────────────────────────────
def tab_trial(trial_id: str, meta_df: pd.DataFrame) -> None:
    result = load_result(trial_id)
    if not result:
        st.error(f"No result file for {trial_id}. Run pipeline first.")
        return

    meta_row = meta_df[meta_df["trial_id"] == trial_id]
    cosmos   = result.get("cosmos_output", {})
    quality  = result.get("quality", {})
    gt_segs  = result.get("gt_gesture_segments", [])
    pd_segs  = result.get("predicted_gesture_segments", [])
    total_f  = quality.get("task_duration_frames", 1)

    skill_gt = result.get("skill_level_gt", "?")
    grs_gt   = result.get("grs_total_gt", 0)
    grs_pred = cosmos.get("predicted_grs_total", quality.get("predicted_grs", 0))
    skill_pred = cosmos.get("predicted_skill_level", quality.get("predicted_skill", "?"))
    gest_acc   = result.get("gesture_accuracy", 0.0)

    # ── Top metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Skill GT",  skill_gt,
              delta=f"Pred: {skill_pred}",
              delta_color="off")
    c2.metric("GRS GT",  grs_gt,
              delta=f"Pred: {int(grs_pred)}",
              delta_color="inverse" if abs(grs_pred - grs_gt) > 3 else "normal")
    c3.metric("Gesture Acc.", f"{gest_acc:.1%}")
    c4.metric("Smoothness",   f"{quality.get('smoothness_score', 0):.3f}")
    c5.metric("Bilateral",    f"{quality.get('bilateral_coordination', 0):.3f}")
    c6.metric("Duration",
              f"{total_f/30:.0f}s",
              delta=f"{total_f} frames")

    # ── Cosmos verdict ───────────────────────────────────────────────────────
    final_result  = cosmos.get("final_result", "unknown")
    fail_reason   = cosmos.get("failure_reason")
    confidence    = cosmos.get("confidence", 0.0)
    result_color  = {"success": "#2ecc71",
                     "partial_success": "#f39c12",
                     "failure": "#e74c3c"}.get(final_result, "#95a5a6")

    st.markdown(
        f"""<div style='background:{result_color}22; border-left:4px solid {result_color};
        padding:10px 16px; border-radius:4px; margin:8px 0'>
        <b>Cosmos Verdict:</b>
        <span style='color:{result_color}; font-size:1.1em'> {final_result.upper()}</span>
        {"  |  <b>Failure reason:</b> " + fail_reason if fail_reason else ""}
        &nbsp;&nbsp;<small>confidence: {confidence:.2f}</small>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Key observations ─────────────────────────────────────────────────────
    obs = cosmos.get("key_observations", [])
    if obs:
        with st.expander("Key Observations", expanded=True):
            for o in obs:
                st.markdown(f"- {o}")

    # ── Kinematic signals plot ───────────────────────────────────────────────
    try:
        kin = load_kinematics(trial_id)
        st.plotly_chart(kinematic_signals_plot(kin, trial_id),
                        use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load kinematics: {e}")

    # ── Gesture timeline ─────────────────────────────────────────────────────
    st.plotly_chart(
        gesture_timeline_plot(gt_segs, pd_segs, total_f, trial_id),
        use_container_width=True,
    )

    # ── GRS radar + quality metrics side by side ──────────────────────────────
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        grs_comp_gt   = result.get("grs_components_gt", {})
        qa            = cosmos.get("quality_assessment", {})
        grs_comp_pred = {
            "respect_for_tissue":    qa.get("smoothness", 0),
            "suture_needle_handling":qa.get("instrument_handling", 0),
            "time_and_motion":       qa.get("time_efficiency", 0),
            "flow_of_operation":     qa.get("flow_of_operation", 0),
            "overall_performance":   qa.get("overall_performance", 0),
            "quality_of_final_product": qa.get("final_product_quality", 0),
        }
        st.plotly_chart(
            grs_radar_plot(grs_comp_gt, grs_comp_pred, trial_id),
            use_container_width=True,
        )

    with col_right:
        st.markdown("#### Quality Metrics")
        qm = {
            "Smoothness Score":      quality.get("smoothness_score", 0),
            "Bilateral Coordination":quality.get("bilateral_coordination", 0),
            "Gesture Accuracy":      result.get("gesture_accuracy", 0),
            "Normalized Jerk (ML)":  quality.get("normalized_jerk_ml", 0),
            "Normalized Jerk (MR)":  quality.get("normalized_jerk_mr", 0),
            "Gripper Event Rate":    quality.get("gripper_event_rate", 0),
            "Speed Efficiency":      quality.get("speed_efficiency", 0),
            "Path Length ML (m)":    quality.get("path_length_ml", 0),
            "Path Length MR (m)":    quality.get("path_length_mr", 0),
        }
        for k, v in qm.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #333'>"
                f"<span style='color:#aaa'>{k}</span>"
                f"<span style='font-weight:bold'>{v:.4f}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Raw Cosmos JSON ───────────────────────────────────────────────────────
    with st.expander("Full Cosmos Output JSON"):
        st.json(cosmos)


# ── Tab: Batch Evaluation ─────────────────────────────────────────────────────
def tab_evaluation(meta_df: pd.DataFrame) -> None:
    summary = load_summary()
    if not summary:
        st.warning(
            "No evaluation summary found. Run:\n\n"
            "```bash\npython jigsaws_pipeline.py --evaluate\n```"
        )
        return

    rows = summary.get("per_trial", [])
    df   = pd.DataFrame(rows)

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trials Evaluated", summary.get("n_samples", 0))
    c2.metric("Mean Gesture Acc.", f"{summary.get('mean_gesture_accuracy', 0):.1%}")
    c3.metric("Mean |GRS Error|",  f"{summary.get('mean_grs_error', 0):.1f} pts")
    c4.metric("Skill Accuracy",    f"{summary.get('skill_accuracy', 0):.1%}")

    st.divider()

    # ── Per-trial table ───────────────────────────────────────────────────────
    st.subheader("Per-Trial Results")
    if not df.empty:
        display_cols = [
            "trial_id", "skill_gt", "grs_gt", "grs_pred", "skill_pred",
            "gesture_accuracy", "smoothness", "bilateral_coordination",
            "final_result", "confidence",
        ]
        avail_cols = [c for c in display_cols if c in df.columns]
        df_show = df[avail_cols].copy()

        def _color_skill(val):
            return f"color: {SKILL_COLORS.get(val, 'white')}"

        def _color_result(val):
            c = {"success": "#2ecc71",
                 "partial_success": "#f39c12",
                 "failure": "#e74c3c"}.get(val, "white")
            return f"color: {c}"

        styled = df_show.style
        if "skill_gt" in avail_cols:
            styled = styled.applymap(_color_skill, subset=["skill_gt"])
        if "skill_pred" in avail_cols:
            styled = styled.applymap(_color_skill, subset=["skill_pred"])
        if "final_result" in avail_cols:
            styled = styled.applymap(_color_result, subset=["final_result"])
        if "gesture_accuracy" in avail_cols:
            styled = styled.background_gradient(
                subset=["gesture_accuracy"], cmap="RdYlGn", vmin=0, vmax=1
            )
        if "smoothness" in avail_cols:
            styled = styled.background_gradient(
                subset=["smoothness"], cmap="RdYlGn", vmin=0, vmax=1
            )
        styled = styled.format({
            "gesture_accuracy":      "{:.3f}",
            "smoothness":            "{:.3f}",
            "bilateral_coordination":"{:.3f}",
            "confidence":            "{:.3f}",
        })
        st.dataframe(styled, use_container_width=True, height=380)

    st.divider()

    # ── Plots row 1 ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(quality_scatter_plot(summary), use_container_width=True)
    with col2:
        st.plotly_chart(kinematic_quality_heatmap(summary), use_container_width=True)

    # ── Gesture composition ───────────────────────────────────────────────────
    st.plotly_chart(gesture_gantt_all(summary), use_container_width=True)

    # ── Skill-level split plots ───────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(skill_distribution_plot(meta_df), use_container_width=True)
    with col4:
        st.plotly_chart(grs_boxplot(meta_df), use_container_width=True)

    # ── Action quality breakdown (knot tying) ─────────────────────────────────
    st.subheader("Action Quality by Skill Level")
    if not df.empty:
        skill_groups = df.groupby("skill_gt").agg(
            mean_smooth=("smoothness", "mean"),
            mean_bilat=("bilateral_coordination", "mean"),
            mean_gest_acc=("gesture_accuracy", "mean"),
            mean_grs_gt=("grs_gt", "mean"),
            mean_grs_pred=("grs_pred", "mean"),
            count=("trial_id", "count"),
        ).reset_index()

        fig_bar = go.Figure()
        metrics = [
            ("mean_smooth",    "Smoothness",          "#2196F3"),
            ("mean_bilat",     "Bilateral Coord.",     "#E91E63"),
            ("mean_gest_acc",  "Gesture Accuracy",     "#4CAF50"),
        ]
        for col, label, color in metrics:
            fig_bar.add_trace(go.Bar(
                x=skill_groups["skill_gt"],
                y=skill_groups[col],
                name=label,
                marker_color=color,
                text=[f"{v:.3f}" for v in skill_groups[col]],
                textposition="outside",
            ))
        fig_bar.update_layout(
            barmode="group",
            title="Mean Quality Metrics by Skill Level",
            xaxis=dict(title="Skill Level", gridcolor="#333"),
            yaxis=dict(title="Score (0-1)", range=[0, 1.15], gridcolor="#333"),
            height=320,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab: Dataset Explorer ─────────────────────────────────────────────────────
def tab_dataset(meta_df: pd.DataFrame) -> None:
    st.subheader("Full JIGSAWS Knot_Tying Dataset")

    # GRS component breakdown
    st.markdown("#### GRS Score Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grs_boxplot(meta_df), use_container_width=True)
    with col2:
        fig_comp = go.Figure()
        for skill, color in SKILL_COLORS.items():
            sub = meta_df[meta_df["skill_level"] == skill]
            if sub.empty:
                continue
            means = [sub[c].mean() for c in GRS_COMPONENTS]
            labels = [c.replace("_", " ").title() for c in GRS_COMPONENTS]
            labels += [labels[0]]
            means  += [means[0]]
            fig_comp.add_trace(go.Scatterpolar(
                r=means, theta=labels,
                fill="toself", name=skill,
                line_color=color, opacity=0.7,
            ))
        fig_comp.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 4], gridcolor="#333"),
                angularaxis=dict(gridcolor="#333"),
                bgcolor="#0e1117",
            ),
            title="Mean GRS Components by Skill Level",
            height=360,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Full meta table
    st.markdown("#### All Trials")
    show_df = meta_df.copy()
    show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]
    styled = show_df.style.background_gradient(
        subset=["Grs Total"], cmap="RdYlGn", vmin=6, vmax=30
    ).applymap(
        lambda v: f"color: {SKILL_COLORS.get(v, 'white')}",
        subset=["Skill Level"]
    )
    st.dataframe(styled, use_container_width=True, height=520)

    # Transcription explorer
    st.subheader("Transcription Explorer")
    available_trans = sorted([p.stem for p in TRANS_DIR.glob("Knot_Tying_*.txt")])
    sel_trial = st.selectbox("Select trial for transcription", available_trans)
    if sel_trial:
        segs = load_transcription(sel_trial)
        df_t = pd.DataFrame(segs)
        if not df_t.empty:
            df_t["duration_frames"] = df_t["end"] - df_t["start"] + 1
            df_t["duration_s"]      = (df_t["duration_frames"] / 30).round(2)
            st.dataframe(df_t, use_container_width=True)

            # Quick timeline
            fig_t = go.Figure()
            for _, row in df_t.iterrows():
                gid   = row["gesture_id"]
                color = GESTURE_COLORS.get(gid, "#607D8B")
                fig_t.add_shape(
                    type="rect",
                    x0=row["start"]/30, x1=row["end"]/30,
                    y0=0.1, y1=0.9,
                    fillcolor=color, opacity=0.8, line=dict(width=0),
                )
                fig_t.add_annotation(
                    x=(row["start"] + row["end"]) / 60,
                    y=0.5,
                    text=gid, showarrow=False,
                    font=dict(size=10, color="white"),
                )
            fig_t.update_layout(
                title=f"Gesture Timeline — {sel_trial}",
                xaxis=dict(title="Time (s)", gridcolor="#333"),
                yaxis=dict(visible=False),
                height=140,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"),
                showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)


# ── Tab: Fusion Architecture ──────────────────────────────────────────────────
def tab_architecture() -> None:
    st.subheader("Fusion Architecture: Kinematic-B + Cosmos")

    st.markdown("""
```
JIGSAWS Knot_Tying Sample (trial_id)
│
├─ kinematics/AllGestures/*.txt  [T × 76 columns @ 30fps]
│    │
│    └─ KinematicFeatureExtractor
│         ├─ ML/MR/SL/SR velocity norms        (8 signals)
│         ├─ ML/MR/SL/SR gripper angles        (4 signals)
│         ├─ Bilateral synchrony               (1 signal)
│         ├─ Jerk (rate of velocity change)    (2 signals)
│         └─ Tool-tip XYZ positions            (6 signals)
│                    │
│                    ├─ Quality metrics (NJ, SPARC, bilateral coord)
│                    └─ Rule-based gesture segmentation
│
├─ video/*.avi (capture1)   [optional]
│    └─ DINOv2 ViT-B/14 (frozen)
│         └─ CLS embeddings [N_keyframes × 768]
│                    │
│                    └─ MLP projection → 64-dim visual context
│
├─ Fusion Layer
│    └─ concat [kinematic_features | dinov2_context]
│         └─ GRU(64, 6-class) → per-frame gesture labels
│              [G1, G11, G12, G13, G14, G15]
│
└─ Cosmos-Reason2-2B
     ├─ Input: gesture sequence + kinematic quality signals
     │         + key video frames (at stage transitions)
     └─ Output: structured JSON
          ├─ task_stages           (gesture sequence)
          ├─ final_result          (success/partial/failure)
          ├─ failure_reason
          ├─ quality_assessment    (GRS-style 0-4 each component)
          ├─ predicted_skill_level (E/I/N)
          ├─ predicted_grs_total   (6-30)
          ├─ key_observations
          └─ confidence            (0-1)
```

### Why this fusion for JIGSAWS?

| Signal Source | What it provides | Role |
|---|---|---|
| **Kinematic data (76-col)** | Position, velocity, rotation, gripper angle for 4 tools | Replaces TAPIR point tracks — already structured trajectories |
| **DINOv2 ViT-B/14** | Spatial patch features, tissue appearance | Captures visual context TAPIR cannot (tissue deformation, needle visibility) |
| **Rule-based / GRU** | Gesture stage segmentation from kinematic transitions | Maps raw signals → structured stage sequence |
| **Cosmos-Reason2-2B** | World-model reasoning on structured signals + frames | Final verdict: quality, success/failure, failure attribution |

### Gesture Vocabulary (Knot_Tying)
| ID | Name | Kinematic Signature |
|---|---|---|
| G1 | Reach for suture | Low velocity, MR approaching needle region |
| G12 | Reach for needle | High MR velocity, gripper transitioning open→close |
| G13 | Position needle | Low velocity, fine adjustments, grippers stable |
| G14 | Push needle through | MR high directional velocity, gripper closed |
| G15 | Pull suture left | ML high velocity, pulling motion, SL active |
| G11 | Release & return | Decreasing velocity, grippers opening |

### Quality Signal Mapping
| Quality Issue | Kinematic Indicator |
|---|---|
| Unstable movement | High normalized jerk (NJ > 1000) |
| Poor coordination | Low bilateral synchrony (< 0.5) |
| Excessive time | task_duration_frames > 1800 (60s) |
| Fumbling | Gripper event rate > 5 events/100 frames |
| Tissue respect | Smooth velocity profiles, low force proxy |
""")

    st.subheader("Model Selection Rationale")
    st.markdown("""
**Why Fusion-B (Kinematic + DINOv2 + GRU + Cosmos) over other fusions?**

- **Fusion A (TAPIR + Cosmos)**: JIGSAWS already provides kinematic data which is *richer* than
  TAPIR tracks — we have full 6-DOF tool states, not just 2D pixel coordinates.

- **Fusion B adapted for JIGSAWS**: The kinematic 76-column data directly maps to what
  TAPIR would produce but in 3D robot workspace coordinates. DINOv2 adds the visual domain
  (tissue appearance, needle-suture contact) that kinematics alone cannot capture.

- **Fusion C (LLoVi/VideoLLaMA2)**: Overkill for single-task episodes of 20-60s.
  Knot_Tying has clear, short, gesture-structured sequences.

- **Fusion D (Full stack)**: Adds RAFT and Depth Anything which are redundant when
  kinematic velocity data is already available at sensor level.

**Key insight**: In JIGSAWS, the kinematic data *is* the structured signal log that other
fusion architectures try to *extract from video*. The fusion role shifts from signal extraction
to signal enrichment (DINOv2 for visual context) + reasoning (Cosmos).
""")


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    meta_df  = load_meta()
    selected = sidebar(meta_df)

    st.title("🔬 JIGSAWS Surgical Skill Assessment")
    st.caption(
        "JHU-ISI Gesture and Skill Assessment Working Set · Knot_Tying Task · "
        "Fusion: Kinematic-B + DINOv2 + Cosmos-Reason2-2B"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Trial Analysis",
        "Batch Evaluation",
        "Dataset Explorer",
        "Architecture",
    ])

    with tab1:
        if selected:
            tab_trial(selected, meta_df)
        else:
            st.info("Run `python jigsaws_pipeline.py --evaluate` to generate results, "
                    "then refresh this page.")

    with tab2:
        tab_evaluation(meta_df)

    with tab3:
        tab_dataset(meta_df)

    with tab4:
        tab_architecture()


if __name__ == "__main__":
    main()
