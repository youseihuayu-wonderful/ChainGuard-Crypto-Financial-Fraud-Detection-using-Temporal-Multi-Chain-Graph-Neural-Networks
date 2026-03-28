# ChainGuard Dashboard

Enterprise-grade fraud detection monitoring platform powered by TH-GNN.

## Quick Start

```bash
# From project root
pip install streamlit plotly networkx
streamlit run dashboard/app.py
```

## Pages

| Page | Description |
|------|-------------|
| Executive Dashboard | KPIs, risk alerts, model ranking |
| Model Performance | Ablation study, baseline comparison, radar chart |
| Transaction Scanner | Interactive fraud risk assessment demo |
| Network Explorer | Graph topology visualization |
| Case Study & Forensics | Detection comparison, fraud patterns |

## Tech Stack

- **Streamlit** — Web framework
- **Plotly** — Interactive charts
- **NetworkX** — Graph operations
- **TH-GNN** — Temporal Heterogeneous GNN (backend model)
