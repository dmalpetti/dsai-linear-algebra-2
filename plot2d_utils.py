"""Small helpers to draw axes, lines and vectors of R^2 with plotly.

Same interface as plot3d_utils.py, one dimension lower; import them with

    from plot2d_utils import new_figure, draw_axes, draw_line, draw_vector
"""

import numpy as np
import plotly.graph_objects as go

# Course color palette (Corbusier palette, Style/style_shared.tex): the drawing functions
# take one of these names, e.g. "orange" or "light orange", instead of a color code
COLORS = {
    "blue":   "#2b70a0", "light blue":   "#89bfc7",
    "green":  "#32804c", "light green":  "#b1cf6c",
    "orange": "#ea6109", "light orange": "#fce3cf",
    "red":    "#9b1737", "light red":    "#edbca8",
    "brown":  "#773f35", "light brown":  "#eec1a7",
    "black":  "#565044", "light black":  "#c7baa8",
}
AXES_COLOR = COLORS["light black"]
TEXT_COLOR = COLORS["black"]


def course_color(name):
    """The color code of a color of the course palette, e.g. course_color("orange")."""
    if name not in COLORS:
        raise ValueError(f"unknown color {name!r}; choose one of: {', '.join(COLORS)}")
    return COLORS[name]


def new_figure(title, size=3):
    """An empty figure: no grid and no box, only the origin O and the title."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                             marker=dict(size=5, color=TEXT_COLOR),
                             text=["O"], textposition="bottom left",
                             textfont=dict(color=TEXT_COLOR),
                             hoverinfo="skip", showlegend=False))

    hidden = dict(visible=False, range=[-size - 0.5, size + 0.5])
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=16)),
                      width=600, height=600, margin=dict(l=0, r=0, t=50, b=0),
                      plot_bgcolor="white",
                      xaxis=hidden,
                      yaxis=dict(scaleanchor="x", scaleratio=1, **hidden))   # equal aspect ratio
    return fig


def draw_axes(fig, size=3):
    """The two coordinate axes through the origin, with an arrow, a label and integer ticks."""
    for k, axis_name in enumerate("xy"):
        e = np.eye(2)[k] * size
        fig.add_trace(go.Scatter(x=[-e[0], e[0]], y=[-e[1], e[1]], mode="lines",
                                 line=dict(color=AXES_COLOR, width=2),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=e[0], y=e[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                           arrowcolor=AXES_COLOR)

        ticks = [t for t in range(-int(size) + 1, int(size)) if t != 0]   # not up to the tip
        marks = np.outer(ticks, np.eye(2)[k])
        fig.add_trace(go.Scatter(x=marks[:, 0], y=marks[:, 1], mode="markers+text",
                                 marker=dict(size=3, color=AXES_COLOR),
                                 text=[str(t) for t in ticks],
                                 textposition="middle left" if axis_name == "y" else "bottom center",
                                 textfont=dict(color=AXES_COLOR, size=10),
                                 hoverinfo="skip", showlegend=False))

        tip = e * 1.12
        fig.add_trace(go.Scatter(x=[tip[0]], y=[tip[1]], mode="text", text=[axis_name],
                                 textfont=dict(color=TEXT_COLOR, size=14),
                                 hoverinfo="skip", showlegend=False))
    return fig


def draw_line(fig, direction, color, name=None, size=3, dash="dash", legend=True):
    """The line through the origin with the given direction; the legend shows its span.

    Without a name, or with legend=False, the line is not added to the legend.
    """
    color = course_color(color)
    d = np.array([float(k) for k in direction])
    d = d / np.linalg.norm(d) * size
    legend = legend and name is not None
    label = f"{name}: ⟨({', '.join(str(k) for k in direction)})⟩"

    fig.add_trace(go.Scatter(x=[-d[0], d[0]], y=[-d[1], d[1]], mode="lines",
                             line=dict(color=color, width=3, dash=dash),
                             name=label, showlegend=legend))
    return fig


def draw_vector(fig, vec, color, name=None, legend=True, tip_label=True):
    """A vector applied at the origin; the legend shows its components.

    Without a name, or with legend=False, the vector is not added to the legend.
    With tip_label=True, the name is also written next to the tip of the vector.
    """
    color = course_color(color)
    x, y = [float(k) for k in vec]
    legend = legend and name is not None
    label = f"{name} = ({', '.join(str(k) for k in vec)})"

    fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines",
                             line=dict(color=color, width=4), name=label,
                             showlegend=legend))
    fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=4,
                       arrowcolor=color)

    if tip_label and name is not None:
        tip = np.array([x, y])
        if np.linalg.norm(tip) > 0:
            tip = tip + 0.15 * tip / np.linalg.norm(tip)   # a little beyond the tip
        fig.add_trace(go.Scatter(x=[tip[0]], y=[tip[1]], mode="text",
                                 text=[name], textfont=dict(color=color, size=14),
                                 hoverinfo="skip", showlegend=False))
    return fig
