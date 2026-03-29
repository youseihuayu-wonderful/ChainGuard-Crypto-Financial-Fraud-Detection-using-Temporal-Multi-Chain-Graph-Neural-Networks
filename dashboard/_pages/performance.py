"""Page 2: Model Performance - Ablation study and baseline comparison."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render(ablation, baseline):
    st.markdown("# 🧪 Model Performance Analysis")
    st.markdown("Comprehensive evaluation of TH-GNN components and comparison with state-of-the-art methods")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ablation Study", "Baseline Comparison", "Detailed Metrics"])

    # ---- Tab 1: Ablation ----
    with tab1:
        st.markdown("### Incremental Component Contribution (M1 → M5)")
        st.markdown(
            "Each model variant adds one component to measure its individual contribution. "
            "**M3 (+Heterogeneous Edges)** provides the largest improvement."
        )

        models = list(ablation.keys())
        names = [ablation[m]["name"] for m in models]
        short = ["M1: GCN", "M2: +Temporal", "M3: +Hetero", "M4: TH-GNN", "M5: +LP"]
        aucs = [ablation[m]["auc_roc"] for m in models]
        f1s = [ablation[m]["f1"] for m in models]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="AUC-ROC", x=short, y=aucs,
            marker_color=["#2196F3", "#2196F3", "#64ffda", "#2196F3", "#2196F3"],
            text=[f"{v:.4f}" for v in aucs], textposition="outside",
            textfont=dict(color="#ccd6f6"),
        ))
        fig.add_trace(go.Bar(
            name="F1 (illicit)", x=short, y=f1s,
            marker_color=["#FF9800", "#FF9800", "#FFB74D", "#FF9800", "#FF9800"],
            text=[f"{v:.4f}" for v in f1s], textposition="outside",
            textfont=dict(color="#ccd6f6"),
        ))

        # Delta annotations
        for i in range(1, len(aucs)):
            delta = aucs[i] - aucs[0]
            fig.add_annotation(
                x=short[i], y=aucs[i] + 0.05,
                text=f"+{delta:.1%}", showarrow=False,
                font=dict(color="#ff5252" if i == 2 else "#8892b0", size=11),
            )

        fig.update_layout(
            barmode="group", height=450,
            yaxis=dict(range=[0, 1.1], title="Score", gridcolor="rgba(255,255,255,0.05)", color="#8892b0"),
            xaxis=dict(color="#ccd6f6"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            legend=dict(orientation="h", y=1.12, x=0.3),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Insight boxes
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="risk-low">'
                '<strong style="color:#64ffda">Key Finding</strong><br>'
                '<span style="color:#ccd6f6">Graph augmentation strategy (temporal k-NN edges) '
                'matters more than model complexity. M3 achieves the best AUC with just '
                'heterogeneous edge modeling.</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="risk-medium">'
                '<strong style="color:#ff9800">Why M4 < M3?</strong><br>'
                '<span style="color:#ccd6f6">Temporal attention adds redundant information '
                'when k-NN edges already encode temporal patterns. '
                'Simpler is better.</span></div>',
                unsafe_allow_html=True,
            )

    # ---- Tab 2: Baseline ----
    with tab2:
        st.markdown("### Comparison with 7 State-of-the-Art Methods")

        results = baseline["results"]
        rows = []
        for key, val in results.items():
            display = {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "gcn_m1": "GCN (M1)",
                "gat": "GAT",
                "graphsage": "GraphSAGE",
                "evolvegcn_h": "EvolveGCN-H",
                "thgnn_m3_ours": "TH-GNN (Ours)",
            }
            rows.append({
                "Method": display.get(key, key),
                "Type": val["type"].title(),
                "AUC-ROC": val["auc_roc"],
                "F1": val["f1"],
                "Precision": val["precision"],
                "Recall": val["recall"],
            })

        df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)

        # Radar chart for top 4
        top4 = df.head(4)
        categories = ["AUC-ROC", "F1", "Precision", "Recall"]

        fig = go.Figure()
        colors_radar = ["#64ffda", "#2196F3", "#9E9E9E", "#9C27B0"]
        for i, (_, row) in enumerate(top4.iterrows()):
            values = [row["AUC-ROC"], row["F1"], row["Precision"], row["Recall"]]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=row["Method"],
                line=dict(color=colors_radar[i], width=2),
                fill='toself',
                fillcolor=f"rgba({','.join(str(int(colors_radar[i].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.1)",
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.1)", color="#8892b0"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#ccd6f6"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            height=450,
            legend=dict(orientation="h", y=-0.1, x=0.1),
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: Detailed Table ----
    with tab3:
        st.markdown("### Full Results Table")
        st.markdown(f"**Data Split:** {baseline['data_split']} | **Seed:** {baseline['seed']} | **Date:** {baseline['date']}")

        df_display = df.copy()
        df_display["AUC-ROC"] = df_display["AUC-ROC"].map("{:.4f}".format)
        df_display["F1"] = df_display["F1"].map("{:.4f}".format)
        df_display["Precision"] = df_display["Precision"].map("{:.4f}".format)
        df_display["Recall"] = df_display["Recall"].map("{:.4f}".format)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Ablation Study Detailed Results")
        abl_rows = []
        for key in ablation:
            m = ablation[key]
            delta = m["auc_roc"] - ablation["M1"]["auc_roc"]
            abl_rows.append({
                "Variant": f"{key}: {m['name']}",
                "AUC-ROC": f"{m['auc_roc']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "Delta vs M1": f"+{delta:.4f}" if delta > 0 else "baseline",
            })
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)
