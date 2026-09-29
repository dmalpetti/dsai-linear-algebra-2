"""Small helpers to draw axes, planes and vectors of R^3 with plotly.

Used by the notebooks of the course; import them with

    from plot3d_utils import new_figure, draw_axes, draw_plane, draw_vector
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
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                               marker=dict(size=3, color=TEXT_COLOR),
                               text=["O"], textposition="top center",
                               textfont=dict(color=TEXT_COLOR),
                               hoverinfo="skip", showlegend=False))

    hidden = dict(visible=False)   # no grid, no box: only the axes we draw ourselves
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=16)),
                      width=750, height=600, margin=dict(l=0, r=0, t=50, b=0),
                      scene=dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                                 camera=dict(eye=dict(x=1.0, y=1.0, z=0.8))))
    return fig


def draw_axes(fig, size=3):
    """The three coordinate axes through the origin, with an arrow, a label and integer ticks."""
    for k, axis_name in enumerate("xyz"):
        e = np.eye(3)[k] * size
        fig.add_trace(go.Scatter3d(x=[-e[0], e[0]], y=[-e[1], e[1]], z=[-e[2], e[2]],
                                   mode="lines", line=dict(color=AXES_COLOR, width=3),
                                   hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Cone(x=[e[0]], y=[e[1]], z=[e[2]], u=[e[0]], v=[e[1]], w=[e[2]],
                              sizemode="absolute", sizeref=0.25, anchor="tip", showscale=False,
                              colorscale=[[0, AXES_COLOR], [1, AXES_COLOR]],
                              hoverinfo="skip", showlegend=False))

        ticks = [t for t in range(-int(size) + 1, int(size)) if t != 0]   # not up to the tip
        marks = np.outer(ticks, np.eye(3)[k])
        fig.add_trace(go.Scatter3d(x=marks[:, 0], y=marks[:, 1], z=marks[:, 2],
                                   mode="markers+text", marker=dict(size=2, color=AXES_COLOR),
                                   text=[str(t) for t in ticks],
                                   textposition="middle left" if axis_name == "z" else "bottom center",
                                   textfont=dict(color=AXES_COLOR, size=9),
                                   hoverinfo="skip", showlegend=False))

        tip = e * 1.15
        fig.add_trace(go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode="text",
                                   text=[axis_name], textfont=dict(color=TEXT_COLOR, size=13),
                                   hoverinfo="skip", showlegend=False))
    return fig


def plane_equation(n):
    """The equation of the plane orthogonal to n, e.g. "x + y - 2z = 0"."""
    terms = []
    for coefficient, variable in zip(n, "xyz"):
        if coefficient == 0:
            continue
        sign = "-" if coefficient < 0 else "+"
        size = "" if abs(coefficient) == 1 else f"{abs(coefficient):g}"   # 2, not 2.0
        terms.append(f"{sign} {size}{variable}")

    equation = " ".join(terms)
    if equation.startswith("+ "):
        equation = equation[2:]
    elif equation.startswith("- "):
        equation = "-" + equation[2:]
    return equation + " = 0"


def draw_plane(fig, n, name="plane", size=3, color="light blue", legend=True):
    """The plane through the origin given by n; the legend shows its equation (unless legend=False).

    n is either the normal vector of the plane, e.g. (1, 1, 1), or a list of two direction
    vectors of the plane, e.g. [v1, v2].
    """
    color = course_color(color)
    n = np.array(n, dtype=float)
    if n.size == 6:                            # two directions: the normal is their cross product
        d1, d2 = n.reshape(2, 3)
        n = np.cross(d1, d2)
        if np.allclose(n, 0):
            raise ValueError("the two direction vectors are parallel: they do not span a plane")
    n = n.ravel()
    b1 = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])   # any direction of the plane
    b2 = np.cross(n, b1)
    b1, b2 = b1 / np.linalg.norm(b1), b2 / np.linalg.norm(b2)

    s = np.linspace(-size + 0.5, size - 0.5, 2)
    S, T = np.meshgrid(s, s)
    plane = S[..., None] * b1 + T[..., None] * b2

    fig.add_trace(go.Surface(x=plane[..., 0], y=plane[..., 1], z=plane[..., 2],
                             colorscale=[[0, color], [1, color]], opacity=0.4,
                             showscale=False, name=f"{name}: {plane_equation(n)}",
                             showlegend=legend))
    return fig


def draw_line(fig, direction, color, name=None, size=3, dash="dash", legend_dash="dot", legend=True):
    """The line through the origin with the given direction; the legend shows its span.

    Without a name, or with legend=False, the line is not added to the legend.

    The line is drawn with `dash`, while the legend sample uses `legend_dash`: the sample is
    very short, and a finer pattern is the only one that reads as a dashed line there.
    """
    color = course_color(color)
    d = np.array([float(k) for k in direction])
    d = d / np.linalg.norm(d) * size
    legend = legend and name is not None
    label = f"{name}: \u27e8({', '.join(str(k) for k in direction)})\u27e9"

    fig.add_trace(go.Scatter3d(x=[-d[0], d[0]], y=[-d[1], d[1]], z=[-d[2], d[2]],
                               mode="lines", line=dict(color=color, width=5, dash=dash),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None],        # legend entry only
                               mode="lines", line=dict(color=color, width=5, dash=legend_dash),
                               name=label, hoverinfo="skip", showlegend=legend))
    return fig


def draw_vector(fig, vec, color, name=None, legend=True, tip_label=True):
    """A vector applied at the origin; the legend shows its components.

    Without a name, or with legend=False, the vector is not added to the legend.
    With tip_label=True, the name is also written next to the tip of the vector.
    """
    color = course_color(color)
    x, y, z = [float(k) for k in vec]
    legend = legend and name is not None
    label = f"{name} = ({', '.join(str(k) for k in vec)})"

    fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines",
                               line=dict(color=color, width=7), name=label,
                               showlegend=legend))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.3, anchor="tip", showscale=False,
                          colorscale=[[0, color], [1, color]], showlegend=False))

    if tip_label and name is not None:
        tip = np.array([x, y, z])
        if np.linalg.norm(tip) > 0:
            tip = tip + 0.15 * tip / np.linalg.norm(tip)   # a little beyond the tip
        fig.add_trace(go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode="text",
                                   text=[name], textfont=dict(color=color, size=13),
                                   hoverinfo="skip", showlegend=False))
    return fig
