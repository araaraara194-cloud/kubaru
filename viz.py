"""グラフ(plotly)。

色の決まり(v1と同じ):
- 混雑率は青1色の濃淡(薄い=空いている、濃い=上限に近い)。
- 上限を超えたコマは「▲」+数値+赤い枠で示す(色だけに頼らない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CRITICAL = "#d03b3b"
INK_DARK = "#0b0b0b"
INK_LIGHT = "#ffffff"


def matrix(rows: pd.DataFrame, area_order: list[str], slot_order: list[str], day: int | None = None):
    """rows(day, slot_name, area_name, total, cap, ratio) → (混雑率, 人数の文字)。day=None なら全日の最大。"""
    L = rows if day is None else rows[rows.day == day]
    L = L[L.cap > 0]
    idx = L.groupby(["area_name", "slot_name"]).ratio.idxmax()
    pick = L.loc[idx.values]
    ratio = pick.pivot(index="area_name", columns="slot_name", values="ratio").reindex(index=area_order, columns=slot_order)
    pick = pick.assign(txt=pick.total.round().astype(int).astype(str) + "/" + pick.cap.astype(int).astype(str) + "人")
    txt = pick.pivot(index="area_name", columns="slot_name", values="txt").reindex(index=area_order, columns=slot_order)
    return ratio, txt


def _label(r) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "休"
    pct = f"{r * 100:.0f}%"
    return f"▲{pct}" if r > 1.0 + 1e-9 else pct


def heatmap(ratio: pd.DataFrame, txt: pd.DataFrame, height: int = 430, show_y: bool = True):
    import plotly.graph_objects as go

    scale = [[i / (len(BLUE_STEPS) - 1), c] for i, c in enumerate(BLUE_STEPS)]
    fig = go.Figure(go.Heatmap(
        z=ratio.clip(upper=1.0).values, x=list(ratio.columns), y=list(ratio.index), zmin=0, zmax=1,
        colorscale=scale, customdata=np.dstack([ratio.values * 100, txt.values]),
        hovertemplate="%{y}<br>%{x}<br>混雑率 %{customdata[0]:.0f}%<br>%{customdata[1]}<extra></extra>",
        xgap=2, ygap=2, showscale=False,
    ))
    for i, area in enumerate(ratio.index):
        for j, slot in enumerate(ratio.columns):
            r = ratio.iloc[i, j]
            ok = not np.isnan(r)
            over = ok and r > 1.0 + 1e-9
            fig.add_annotation(x=slot, y=area, text=f"<b>{_label(r)}</b>" if over else _label(r), showarrow=False,
                               font=dict(size=12, color=INK_LIGHT if ok and min(r, 1.0) > 0.5 else INK_DARK))
            if over:
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                              line=dict(color=CRITICAL, width=2.5), xref="x", yref="y")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed", showticklabels=show_y), xaxis=dict(side="top"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
