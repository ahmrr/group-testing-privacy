import sys
import math
import numpy as np
import altair as alt
import streamlit as st
from pathlib import Path
from pandas import DataFrame
from dataclasses import dataclass

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from src import gtdp


@dataclass
class AxisParams:
    col: str
    fmt: str | None = None
    scale: alt.Scale | None = None
    title: str | None = None
    tooltip: str | None = None
    tooltip_fmt: str | None = None


def _choose_format(arr, default=".2f") -> str:
    max_val = np.nanmax(np.abs(arr))
    min_val = np.nanmin(np.abs(arr[arr != 0])) if np.any(arr != 0) else 0
    if max_val >= 1e4 or (0 < min_val < 1e-3):
        return ".1e"
    return default


def plot_interactive_line(
    df,
    x_axis,
    y_axis,
    chart_title=None,
    chart_layers=[],
    # x_col,
    # y_col,
    # x_title=None,
    # y_title=None,
    # x_tooltip=None,
    # y_tooltip=None,
    # tooltip_format=".2f",
):
    """Plot an interactive Altair line chart"""

    if x_axis.fmt == "adapt":
        x_axis.fmt = _choose_format(df[x_axis.col])
    if y_axis.fmt == "adapt":
        y_axis.fmt = _choose_format(df[y_axis.col])
    if x_axis.tooltip_fmt == "adapt":
        x_axis.tooltip_fmt = _choose_format(df[x_axis.col])
    if y_axis.tooltip_fmt == "adapt":
        y_axis.tooltip_fmt = _choose_format(df[y_axis.col])

    nearest = alt.selection_point(
        nearest=True,
        on="pointerover",
        fields=[x_axis.col],
        empty=False,
        clear="mouseout",
    )

    x_enc = alt.X(
        f"{x_axis.col}:Q",
        title=x_axis.title or x_axis.col,
        axis=alt.Axis(format=x_axis.fmt or alt.Undefined),
        scale=x_axis.scale or alt.Scale(zero=False, nice=True),
    )
    y_enc = alt.Y(
        f"{y_axis.col}:Q",
        title=y_axis.title or y_axis.col,
        axis=alt.Axis(format=y_axis.fmt or alt.Undefined),
        scale=y_axis.scale or alt.Scale(zero=False, nice=True),
    )
    tooltip_enc = [
        alt.Tooltip(
            f"{x_axis.col}:Q",
            title=x_axis.tooltip or x_axis.col,
            format=x_axis.tooltip_fmt or alt.Undefined,
        ),
        alt.Tooltip(
            f"{y_axis.col}:Q",
            title=y_axis.tooltip or y_axis.col,
            format=y_axis.tooltip_fmt or alt.Undefined,
        ),
    ]

    base = alt.Chart(df)

    endpoints = (
        alt.Chart(
            df[df[x_axis.col].isin([df[x_axis.col].min(), df[x_axis.col].max()])]
        )  # very goofy
        .mark_point(filled=True, size=25, opacity=1)
        .encode(x=x_enc, y=y_enc, tooltip=tooltip_enc)
    )
    line = base.mark_line().encode(
        x=x_enc,
        y=alt.Y(f"{y_axis.col}:Q", title=y_axis.title or y_axis.col),
    )
    points = (
        base.mark_point(filled=True, size=50, opacity=0)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=tooltip_enc,
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        )
        .add_params(nearest)
    )
    rules = (
        base.mark_rule(color="gray")
        .encode(
            x=x_enc,
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        )
        .transform_filter(nearest)
    )

    chart = endpoints + line + points + rules
    if chart_title:
        chart = chart.properties(title=chart_title, width="container").configure_title(
            anchor="middle", fontSize=20
        )
    for layer in chart_layers:
        chart += layer.encode(x=x_enc, y=y_enc)

    return chart


st.header("Private and Optimal Group Testing Design", anchor=False)

st.write(
    r"""
This is a tool to help you design optimal group testing experiments for prevalence estimation that inherently provide a chosen level of $\varepsilon$-differential privacy.

Given the sensitivity $S_e$ and specificity $S_p$ of a particular diagnostic test, and given a general idea of the disease prevalence $p$, we provide the optimal group size and artificial noise that guarantee $\varepsilon$-differential privacy with the smallest possible variance in the prevalence estimate.
"""
)

with st.expander("**Fixed Parameters**", expanded=True), st.form("params"):
    col1, col2 = st.columns(2)

    # TODO: deal with incorrect stacking on mobile
    Se = col1.slider(
        r"Diagnostic Test Sensitivity $(S_e)$",
        min_value=0.5,
        max_value=1.0,
        value=0.99,
        step=0.01,
        format="%0.2f",
        help="The true positive rate of the diagnostic test.",
    )
    Sp = col2.slider(
        r"Diagnostic Test Specificity $(S_p)$",
        min_value=0.5,
        max_value=1.0,
        value=0.99,
        step=0.01,
        format="%0.2f",
        help="The true negative rate of the diagnostic test.",
    )
    p = col1.number_input(
        r"Prevalence of Positives $(p)$",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%0.3f",
        help="The speculated proportion of positive individuals in the population.",
    )
    ε_star = col2.number_input(
        r"Desired Epsilon $(\varepsilon)$",
        min_value=0.0,
        value=0.1,
        step=0.001,
        format="%0.3f",
        help="The desired level of differential privacy.",
    )

    submitted = st.form_submit_button(
        label="Submit", on_click=lambda: None, type="primary"
    )

if submitted:
    st.subheader("Inherent Differential Privacy", anchor=False)

    #  Following this, we show what to do after observing the data to achieve a particular level of differential privacy.
    st.write(
        r"""
Here is the inherent amount of differential privacy (without adding artificial noise after observing the data) that your setup can provide, as a function of some pool sizes $c$.             

Inherent privacy is a particularly strong notion of privacy, since it does not require anyone to observe the data at all to guarantee privacy."""
    )

    pool_sizes = 1 + np.arange(1, 100)
    privacy = gtdp.formulas.epsilon(p, pool_sizes, Se, Sp)
    privacy_chart = plot_interactive_line(
        df=DataFrame(
            {
                "c": pool_sizes,
                "eps": privacy,
            }
        ),
        x_axis=AxisParams(col="c", fmt="d", title="Pool Size", tooltip="Pool size:"),
        y_axis=AxisParams(
            col="eps",
            title="Differential Privacy Level",
            tooltip="Privacy level:",
            tooltip_fmt="adapt",
        ),
        chart_title="Privacy for Different Pool Sizes",
    )
    st.altair_chart(privacy_chart, use_container_width=True)
    # st.line_chart(data, x_label=r"Pool Size (c)", y_label=r"Privacy Level (ε)")

    st.subheader(
        r"Optimally Achieving $\varepsilon$-Differential Privacy", anchor=False
    )

    J = 100  # TODO: custom J and fixed N

    c_star = gtdp.formulas.optimal_pool_size(p, ε_star)
    optimal_c = gtdp.formulas.round_pool_size(p, c_star, J, ε_star, fixed_N=None)
    Se_star, Sp_star = gtdp.formulas.optimal_accuracy(p, optimal_c, ε_star)

    # TODO: find
    gamma_1 = 0.5
    gamma_2 = 0.3

    # The optimal sensitivity and specificity are $(S_e, S_p) = ({Se_star}, {Sp_star})$. Do the following after observing the pooled test results:
    st.write(
        rf"""
In short, to achieve $\varepsilon={math.floor(ε_star * 10**3) / 10**3}$ differential privacy with lowest variance in the estimator:

 1. Choose a **pool size** of $c={optimal_c}$
 2. Flip **positive** pooled test results with probability $\Gamma_1 = {gamma_1}$
 3. Flip **negative** pooled test results with probability $\Gamma_2 = {gamma_2}$
"""
    )

    c_neighborhood = np.arange(2, 2 * max(int(optimal_c), 100))
    optimal_c_df = DataFrame(
        {
            "c": [optimal_c],
            "var": [gtdp.formulas.optimal_pool_size_variance(p, optimal_c, J, ε_star)],
            "label": ["Minimum Variance"],
        }
    )
    optimal_c_point = alt.Chart(optimal_c_df).mark_point(
        filled=True, size=100, opacity=1, color="purple", shape="triangle"
    )
    optimal_c_label = (
        alt.Chart(optimal_c_df)
        .mark_text(align="center", dx=5, dy=20, fontSize=14)
        .encode(text="label")
    )
    pool_size_variance = plot_interactive_line(
        df=DataFrame(
            {
                "c": c_neighborhood,
                "var": gtdp.formulas.optimal_pool_size_variance(
                    p, c_neighborhood, J, ε_star
                ),
            }
        ),
        x_axis=AxisParams(col="c", fmt="d", title="Pool Size", tooltip="Pool size:"),
        y_axis=AxisParams(
            col="var",
            fmt="adapt",
            scale=alt.Scale(zero=False, nice=True, type="log"),
            title="Variance of Prevalence Estimate",
            tooltip="Variance:",
            tooltip_fmt="adapt",
        ),
        chart_title="Variance for Pool Sizes around the Optimum",
        chart_layers=[optimal_c_point, optimal_c_label],
    )
    st.altair_chart(pool_size_variance, use_container_width=True)
