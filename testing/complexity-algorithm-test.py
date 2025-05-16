import numpy as np
import plotly.graph_objs as go

# Define F and N ranges
F_values = np.arange(2, 16)
N_values = np.arange(2, 16)
F, N = np.meshgrid(F_values, N_values)

# Calculate complexities
complexity_exponential = 2 ** (F * N)        # MIP
complexity_polynomial = (F ** 2) * (N ** 3)  # EF-TTC

# Equality curve: where polynomial ≈ exponential
equality_F = []
equality_N = []
equality_Z = []

for f in F_values:
    for n in N_values:
        poly = (f ** 2) * (n ** 3)
        exp = 2 ** (f * n)
        ratio = poly / exp
        if 0.8 < ratio < 1.2:  # Approximate equality zone
            equality_F.append(f)
            equality_N.append(n)
            equality_Z.append(poly)

# EF-TTC surface (blue)
surface_polynomial = go.Surface(
    z=complexity_polynomial,
    x=F,
    y=N,
    colorscale='Blues',
    opacity=0.95,
    name='EF-TTC (Polynomial)',
    showscale=False
)

# MIP surface (orange), with reduced opacity
surface_exponential = go.Surface(
    z=complexity_exponential,
    x=F,
    y=N,
    colorscale='Oranges',
    opacity=0.4,
    name='MIP (Exponential)',
    showscale=False
)

# Equality curve (black line)
equality_curve = go.Scatter3d(
    x=equality_F,
    y=equality_N,
    z=equality_Z,
    mode='lines+markers',
    line=dict(color='black', width=4),
    marker=dict(size=4),
    name='EF-TTC = MIP'
)

# Layout
layout = go.Layout(
    title='EF-TTC vs MIP: Complexity with Equality Curve',
    scene=dict(
        xaxis=dict(title='F (Functions)'),
        yaxis=dict(title='N (Nodes)'),
        zaxis=dict(title='Complexity', type='log'),
        camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))  # elevated angle
    ),
    legend=dict(
        x=0.75,
        y=0.95,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1
    )
)

# Create and display figure
fig = go.Figure(data=[surface_polynomial, surface_exponential, equality_curve], layout=layout)
import plotly.io as pio
pio.show(fig)
