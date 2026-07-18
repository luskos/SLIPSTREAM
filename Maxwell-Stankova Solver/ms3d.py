import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dataclasses import dataclass
from collections import deque
import time
import random

# ============================================================
# OPTIMIZED PARAMETERS
# ============================================================

@dataclass
class WarpParams:
    v: float = 2.5
    R0: float = 1.2
    Aw: float = 0.08
    wW: float = 0.15
    Ad: float = 0.05
    wD: float = 0.15
    epsBD: float = 0.02
    wBD: float = 0.584056828
    bBI: float = 2.319271223957
    grid_size: int = 48
    L: float = 5.0
    dt: float = 0.001

# ============================================================
# LEAP-MORT BYPASS ENGINE WITH PERTURBATIONS
# ============================================================

class LEAPMORTBypass:
    def __init__(self, params: WarpParams):
        self.p = params
        self.grid = np.linspace(-params.L, params.L, params.grid_size)
        self.dx = self.grid[1] - self.grid[0]
        
        # Pre-compute grid
        self.x, self.y, self.z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2 + 0.001**2)
        
        self.Phi = np.zeros((params.grid_size, params.grid_size, params.grid_size))
        
        # LEAP-MORT state
        self.frequency_buffer = np.zeros((32, 32))
        self.photorefractive_memory = np.zeros((32, 32))
        self.mort_angle = 0.0
        self.rpe_phase = 0.0
        self.buffer_fill = 0.0
        self.pfb_triggered = False
        
        # Stability metrics
        self.total_energy = 0.0
        self.ANEC = 0.0
        self.min_NEC = 0.0
        self.is_stable = True
        self.horizon_detected = False
        
        # Energy tracking
        self.energy_target = -0.189028
        self.energy_history = deque(maxlen=100)
        self.anec_history = deque(maxlen=100)
        
        # Perturbation tracking
        self.perturbation_active = False
        self.perturbation_type = "none"
        self.perturbation_time = 0.0
        self.perturbation_magnitude = 0.0
        self.stabilization_count = 0
        
    def compute_phi(self, t: float) -> np.ndarray:
        r = self.r
        x = self.x
        
        # Base field
        Phi_WH = self.p.Aw * (1 - self.p.R0/r) * np.exp(-(r - self.p.R0)**2 / self.p.wW**2)
        Phi_Drive = self.p.Ad * (x - self.p.v * t) * np.exp(-(x - self.p.v * t)**2 / self.p.wD**2)
        Phi_BD = self.p.epsBD * np.exp(-r**2 / self.p.wBD**2)
        Phi_BI = -1/self.p.bBI * np.sqrt(1 + (r/self.p.R0)**2)
        
        Phi = Phi_WH + Phi_Drive + Phi_BD + Phi_BI
        
        # Apply perturbation if active
        if self.perturbation_active:
            Phi = self.apply_perturbation(Phi, t)
        
        return Phi
    
    def apply_perturbation(self, Phi: np.ndarray, t: float) -> np.ndarray:
        """Apply artificial perturbation to the field"""
        center = self.p.grid_size // 2
        x = self.x
        y = self.y
        z = self.z
        r = self.r
        
        if self.perturbation_type == "spike":
            # Localized energy spike
            x0 = 0.0
            y0 = 1.5
            z0 = 0.0
            sigma = 0.5
            perturbation = self.perturbation_magnitude * np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / sigma**2)
            Phi += perturbation
            
        elif self.perturbation_type == "wave":
            # Traveling wave perturbation
            k = 2.0
            perturbation = self.perturbation_magnitude * np.sin(k * (x - 0.5 * t)) * np.exp(-r**2 / 4.0)
            Phi += perturbation
            
        elif self.perturbation_type == "asymmetry":
            # Asymmetric deformation
            perturbation = self.perturbation_magnitude * np.sin(2 * np.arctan2(y, x)) * np.exp(-(r - 2.0)**2 / 1.0)
            Phi += perturbation
            
        elif self.perturbation_type == "ring":
            # Ring perturbation
            r0 = 2.5
            perturbation = self.perturbation_magnitude * np.exp(-(r - r0)**2 / 0.5) * np.cos(4 * np.arctan2(y, x))
            Phi += perturbation
            
        elif self.perturbation_type == "random":
            # Random noise (small amplitude)
            noise = np.random.normal(0, self.perturbation_magnitude * 0.01, Phi.shape)
            Phi += noise
            
        elif self.perturbation_type == "shock":
            # Shock wave
            x0 = 3.0 * np.sin(t * 0.5)
            perturbation = self.perturbation_magnitude * np.exp(-((x - x0)**2 + y**2 + z**2) / 0.3)
            Phi += perturbation
            
        return Phi
    
    def trigger_perturbation(self, ptype: str, magnitude: float = 0.5):
        """Trigger an artificial perturbation"""
        self.perturbation_active = True
        self.perturbation_type = ptype
        self.perturbation_magnitude = magnitude
        self.perturbation_time = time.time()
        print(f"🔴 Perturbation triggered: {ptype} (magnitude: {magnitude})")
    
    def clear_perturbation(self):
        """Clear active perturbation"""
        self.perturbation_active = False
        self.perturbation_type = "none"
        self.perturbation_magnitude = 0.0
        print("✅ Perturbation cleared")
    
    def compute_energy_conditions(self, Phi: np.ndarray) -> dict:
        h = self.dx
        grad = np.gradient(Phi, h, edge_order=2)
        grad_norm2 = grad[0]**2 + grad[1]**2 + grad[2]**2
        
        laplacian = (np.gradient(grad[0], h, edge_order=2)[0] + 
                     np.gradient(grad[1], h, edge_order=2)[1] + 
                     np.gradient(grad[2], h, edge_order=2)[2])
        
        G_tt = np.exp(4 * Phi) * (2 * laplacian - 3 * grad_norm2)
        T_00 = G_tt / (8 * np.pi)
        
        Phi_x = grad[0]
        Phi_xx = np.gradient(Phi_x, h, edge_order=2)[0]
        G_xx = (-2 * laplacian + grad_norm2) + 2 * Phi_x**2 - 2 * Phi_xx
        T_xx = G_xx / (8 * np.pi)
        
        NEC = T_00 + T_xx
        
        return {'T00': T_00, 'Txx': T_xx, 'NEC': NEC}
    
    def leap_mort_pipeline(self, field: np.ndarray, t: float) -> np.ndarray:
        center = self.p.grid_size // 2
        slice_2d = field[center, :, :]
        
        h, w = slice_2d.shape
        step_h = max(1, h // 32)
        step_w = max(1, w // 32)
        sample = slice_2d[::step_h, ::step_w][:32, :32]
        
        self.frequency_buffer += np.abs(sample)
        
        max_possible = np.max(self.frequency_buffer) * 32 * 32
        if max_possible > 0:
            self.buffer_fill = np.sum(self.frequency_buffer) / max_possible
        else:
            self.buffer_fill = 0
        
        if self.buffer_fill > 0.95:
            self.pfb_triggered = True
            coherent = np.ones_like(self.frequency_buffer) * np.mean(self.frequency_buffer)
            self.frequency_buffer *= 0
            
            coherent *= 1200.0
            
            self.mort_angle += 0.1 * t
            for i in range(32):
                for j in range(32):
                    angle = np.arctan2(j - 16, i - 16)
                    coherent[i, j] *= (1 + 0.5 * np.cos(2*np.pi * (i/16 + j/16 + t)))
            
            laplacian = np.zeros_like(coherent)
            laplacian[1:-1, 1:-1] = (coherent[2:, 1:-1] + coherent[:-2, 1:-1] +
                                      coherent[1:-1, 2:] + coherent[1:-1, :-2] -
                                      4 * coherent[1:-1, 1:-1])
            self.photorefractive_memory += (0.1 * coherent - self.photorefractive_memory / 0.1) * 0.01
            self.photorefractive_memory += 0.01 * laplacian * 0.01
            
            self.rpe_phase += 0.1 * t
            for n in range(1, 4):
                phase = 2 * np.pi * (np.arange(32) / 32) * n + self.rpe_phase
                coherent += 0.05 * np.sin(phase[:, None] + phase[None, :])
            
            return coherent
        else:
            self.pfb_triggered = False
            return sample
    
    def step(self, t: float) -> dict:
        start_time = time.time()
        
        self.Phi = self.compute_phi(t)
        energy = self.compute_energy_conditions(self.Phi)
        stabilized_field = self.leap_mort_pipeline(self.Phi, t)
        
        center = self.p.grid_size // 2
        self.total_energy = np.sum(energy['T00']) * self.dx**3
        self.ANEC = np.sum(energy['NEC'][center, :, center]) * self.dx
        self.min_NEC = np.min(energy['NEC'])
        
        g_tt = np.exp(4 * self.Phi) * (1 - self.p.v**2 * np.exp(-4 * self.Phi))
        self.horizon_detected = np.any(np.abs(g_tt) < 0.01)
        
        was_stable = self.is_stable
        self.is_stable = (self.ANEC > 0) and (self.total_energy > -10) and not self.horizon_detected
        
        # Count stabilizations
        if not was_stable and self.is_stable:
            self.stabilization_count += 1
            print(f"🟢 STABILIZED! (Count: {self.stabilization_count})")
        
        # Auto-stabilization feedback
        if not self.is_stable:
            energy_error = (self.total_energy - self.energy_target) / abs(self.energy_target)
            self.p.Ad *= (1 + 0.001 * energy_error)
            self.p.Aw *= (1 + 0.001 * energy_error * 0.5)
            self.p.Ad = np.clip(self.p.Ad, 0.01, 0.1)
            self.p.Aw = np.clip(self.p.Aw, 0.01, 0.15)
            self.p.v = np.clip(self.p.v, 1.1, 3.0)
        
        self.energy_history.append(self.total_energy)
        self.anec_history.append(self.ANEC)
        
        elapsed = time.time() - start_time
        
        return {
            'total_energy': float(self.total_energy),
            'ANEC': float(self.ANEC),
            'min_NEC': float(self.min_NEC),
            'is_stable': bool(self.is_stable),
            'horizon_detected': bool(self.horizon_detected),
            'step_time_ms': float(elapsed * 1000),
            'ad': float(self.p.Ad),
            'aw': float(self.p.Aw),
            'buffer_fill': float(self.buffer_fill * 100),
            'pfb_triggered': bool(self.pfb_triggered),
            'Phi': self.Phi.copy().tolist(),
            'T00': energy['T00'].copy().tolist(),
            'NEC': energy['NEC'].copy().tolist(),
            'perturbation_active': bool(self.perturbation_active),
            'perturbation_type': self.perturbation_type,
            'stabilization_count': int(self.stabilization_count)
        }

# ============================================================
# DASH APPLICATION
# ============================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Stankova Slipstream - With Perturbations"

# Global state
params = WarpParams()
solver = LEAPMORTBypass(params)
t = 0.0
running = False
first_run = True

def validate_diag(diag):
    """Validate that diag is a proper dictionary with required keys"""
    if diag is None:
        return None
    if not isinstance(diag, dict):
        return None
    required_keys = ['Phi', 'T00', 'NEC', 'total_energy', 'ANEC', 'min_NEC', 
                     'is_stable', 'horizon_detected', 'buffer_fill', 'pfb_triggered',
                     'ad', 'aw', 'step_time_ms', 'perturbation_active', 'perturbation_type']
    for key in required_keys:
        if key not in diag:
            return None
    return diag

def create_visualization(diag):
    """Create visualization with perturbation indicators"""
    diag = validate_diag(diag)
    if diag is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for solver data...<br>Click 'Run' to start",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(template='plotly_dark', height=700, margin=dict(l=0, r=0, t=0, b=0))
        return fig
    
    # Convert back to numpy arrays
    Phi = np.array(diag['Phi'])
    T00 = np.array(diag['T00'])
    NEC = np.array(diag['NEC'])
    
    center = Phi.shape[0] // 2
    
    # Extract slices
    slice_phi = Phi[center, :, :]
    slice_t00 = T00[center, :, :]
    slice_nec = NEC[center, :, :]
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Φ Field (y=0)', 'Energy Density T₀₀ (y=0)',
                        'NEC Distribution (y=0)', 'Energy & ANEC History'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    x_axis = np.linspace(-params.L, params.L, params.grid_size)
    
    # 1. Phi slice with perturbation indicator
    fig.add_trace(
        go.Heatmap(
            z=slice_phi,
            x=x_axis,
            y=x_axis,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title='Φ', x=0.45, len=0.4)
        ),
        row=1, col=1
    )
    
    # Add perturbation indicator on the plot
    if diag['perturbation_active']:
        fig.add_annotation(
            text=f"⚠ PERTURBATION: {diag['perturbation_type'].upper()}",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14, color="red"),
            row=1, col=1
        )
    
    # 2. Energy density slice
    fig.add_trace(
        go.Heatmap(
            z=slice_t00,
            x=x_axis,
            y=x_axis,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='T₀₀', x=0.45, len=0.4)
        ),
        row=1, col=2
    )
    
    # 3. NEC slice
    fig.add_trace(
        go.Heatmap(
            z=slice_nec,
            x=x_axis,
            y=x_axis,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title='NEC', x=0.45, len=0.4)
        ),
        row=2, col=1
    )
    
    # 4. Energy history plot
    energy_data = list(solver.energy_history)
    anec_data = list(solver.anec_history)
    
    if len(energy_data) > 0:
        time_data = np.linspace(0, len(energy_data) * params.dt, len(energy_data))
        
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=energy_data,
                mode='lines',
                name='Total Energy',
                line=dict(color='cyan', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=anec_data,
                mode='lines',
                name='ANEC',
                line=dict(color='yellow', width=2)
            ),
            row=2, col=2
        )
        
        # Add target line
        fig.add_hline(y=solver.energy_target, line_dash="dash", line_color="red", 
                      annotation_text="Target", row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="green", 
                      annotation_text="ANEC=0", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=True,
        legend=dict(x=0.5, y=1.0, orientation='h'),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_xaxes(title_text='x', row=2, col=1)
    fig.update_xaxes(title_text='Time (s)', row=2, col=2)
    
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=2)
    fig.update_yaxes(title_text='y', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=2)
    
    return fig

def create_hud(diag):
    """Create HUD display with perturbation info"""
    diag = validate_diag(diag)
    if diag is None:
        return html.Div("Waiting for solver...")
    
    stability = "✅ STABLE" if diag['is_stable'] else "❌ UNSTABLE"
    horizon = "✅ NONE" if not diag['horizon_detected'] else "⚠ DETECTED"
    pfb = "● ACTIVE" if diag['pfb_triggered'] else "○ WAITING"
    
    color = "success" if diag['is_stable'] else "danger"
    
    # Perturbation status
    if diag['perturbation_active']:
        pert_status = html.Span(f"⚠ {diag['perturbation_type'].upper()}", className="text-danger")
    else:
        pert_status = html.Span("✓ NONE", className="text-success")
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("STANKOVA SLIPSTREAM", className="text-primary"),
                    html.H6(f"t = {t:.3f} s", className="text-muted"),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span("Stability: ", className="text-muted"),
                        html.Span(stability, className=f"text-{color}")
                    ]),
                    html.Div([
                        html.Span("Horizon: ", className="text-muted"),
                        html.Span(horizon, className="text-warning" if diag['horizon_detected'] else "text-success")
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span("Total Energy: ", className="text-muted"),
                        html.Span(f"{diag['total_energy']:.4f}", className="text-info")
                    ]),
                    html.Div([
                        html.Span("ANEC: ", className="text-muted"),
                        html.Span(f"{diag['ANEC']:.4f}", className="text-info")
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span("Min NEC: ", className="text-muted"),
                        html.Span(f"{diag['min_NEC']:.4f}", className="text-info")
                    ]),
                    html.Div([
                        html.Span("Stabilizations: ", className="text-muted"),
                        html.Span(f"{diag['stabilization_count']}", className="text-success")
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span("PFB: ", className="text-muted"),
                        html.Span(pfb, className="text-warning" if diag['pfb_triggered'] else "text-muted")
                    ]),
                    html.Div([
                        html.Span("Buffer: ", className="text-muted"),
                        dbc.Progress(value=diag['buffer_fill'], max=100, color="info", 
                                    style={'height': '8px', 'width': '80px'}, className="d-inline-block ms-1")
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span("Perturbation: ", className="text-muted"),
                        pert_status
                    ]),
                    html.Div([
                        html.Span("Ad: ", className="text-muted"),
                        html.Span(f"{diag['ad']:.4f}", className="text-warning"),
                        html.Span(" Aw: ", className="text-muted"),
                        html.Span(f"{diag['aw']:.4f}", className="text-warning"),
                    ]),
                ], width=2),
            ])
        ])
    ], className="mb-3")

# ============================================================
# DASH LAYOUT
# ============================================================

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stankova Slipstream", className="text-center text-primary my-3"),
            html.H4("Real-Time Field Solver with LEAP-MORT Stabilization", 
                   className="text-center text-muted mb-3"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("▶ Run", id="run-button", color="success", className="me-2"),
                dbc.Button("⏸ Pause", id="pause-button", color="warning", className="me-2"),
                dbc.Button("⟳ Reset", id="reset-button", color="danger", className="me-2"),
            ], className="mb-3"),
            dbc.Checklist(
                options=[{"label": " Auto-Stabilization", "value": "auto"}],
                value=["auto"],
                id="auto-stabilize",
                switch=True,
                className="ms-3"
            ),
            html.Span(id="fps-display", className="ms-3 text-info")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔴 PERTURBATION CONTROLS"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("💥 Spike", id="pert-spike", color="danger", size="sm", className="me-1"),
                            dbc.Button("🌊 Wave", id="pert-wave", color="warning", size="sm", className="me-1"),
                            dbc.Button("🌀 Asymmetry", id="pert-asym", color="info", size="sm", className="me-1"),
                            dbc.Button("⭕ Ring", id="pert-ring", color="secondary", size="sm", className="me-1"),
                            dbc.Button("🎲 Random", id="pert-random", color="primary", size="sm", className="me-1"),
                            dbc.Button("💫 Shock", id="pert-shock", color="purple", size="sm", className="me-1"),
                            dbc.Button("✖ Clear", id="pert-clear", color="light", size="sm"),
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Magnitude: ", className="text-muted"),
                                dcc.Slider(
                                    id="pert-magnitude",
                                    min=0.1, max=2.0, step=0.1, value=0.5,
                                    marks={0.1: '0.1', 1.0: '1.0', 2.0: '2.0'},
                                    className="mt-2"
                                )
                            ])
                        ], width=6)
                    ])
                ])
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="visualization", style={'height': '750px'}, config={'responsive': True})
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="hud-display")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="status-display", className="text-muted text-center mt-2")
        ])
    ]),
    
    dcc.Interval(id="update-interval", interval=200, n_intervals=0),
    dcc.Store(id="solver-state"),
    
], fluid=True, className="bg-dark text-light min-vh-100")

# ============================================================
# DASH CALLBACKS
# ============================================================

@app.callback(
    [Output("solver-state", "data"),
     Output("status-display", "children"),
     Output("fps-display", "children")],
    [Input("run-button", "n_clicks"),
     Input("pause-button", "n_clicks"),
     Input("reset-button", "n_clicks"),
     Input("update-interval", "n_intervals"),
     Input("pert-spike", "n_clicks"),
     Input("pert-wave", "n_clicks"),
     Input("pert-asym", "n_clicks"),
     Input("pert-ring", "n_clicks"),
     Input("pert-random", "n_clicks"),
     Input("pert-shock", "n_clicks"),
     Input("pert-clear", "n_clicks")],
    [State("auto-stabilize", "value"),
     State("solver-state", "data"),
     State("pert-magnitude", "value")]
)
def update_solver(run_clicks, pause_clicks, reset_clicks, n_intervals,
                  spike_clicks, wave_clicks, asym_clicks, ring_clicks,
                  random_clicks, shock_clicks, clear_clicks,
                  auto_stabilize, state_data, magnitude):
    global t, running, solver, params, first_run
    
    ctx = callback_context
    if not ctx.triggered:
        if first_run:
            first_run = False
            diag = solver.step(0.0)
            return diag, "Ready", ""
        return state_data, "Ready", ""
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle perturbation buttons
    if trigger in ["pert-spike", "pert-wave", "pert-asym", "pert-ring", "pert-random", "pert-shock"]:
        ptype = trigger.replace("pert-", "")
        solver.trigger_perturbation(ptype, magnitude)
        return state_data, f"⚠ Perturbation: {ptype}", ""
    
    if trigger == "pert-clear":
        solver.clear_perturbation()
        return state_data, "Perturbation cleared", ""
    
    if trigger == "reset-button":
        t = 0.0
        solver = LEAPMORTBypass(params)
        running = False
        diag = solver.step(t)
        return diag, "Reset to initial state", ""
    
    if trigger == "run-button":
        running = True
        return state_data, "Running...", ""
    
    if trigger == "pause-button":
        running = False
        return state_data, "Paused", ""
    
    if trigger == "update-interval" and running:
        t += params.dt
        diag = solver.step(t)
        
        fps = 1000 / diag['step_time_ms'] if diag['step_time_ms'] > 0 else 0
        status = f"t = {t:.3f}s | Grid: {params.grid_size}³ | Stabilizations: {diag['stabilization_count']}"
        fps_text = f"⚡ {fps:.1f} FPS"
        return diag, status, fps_text
    
    return state_data, "Ready", ""

@app.callback(
    Output("visualization", "figure"),
    [Input("solver-state", "data")]
)
def update_visualization(diag):
    return create_visualization(diag)

@app.callback(
    Output("hud-display", "children"),
    [Input("solver-state", "data")]
)
def update_hud(diag):
    return create_hud(diag)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("STANKOVA SLIPSTREAM - REAL-TIME FIELD SOLVER")
    print("With Artificial Perturbations for Demonstration")
    print("="*70)
    print(f"\nPARAMETERS:")
    print(f"  Velocity: {params.v} c")
    print(f"  Throat radius: {params.R0} m")
    print(f"  Wormhole deformation: {params.Aw}")
    print(f"  Drive deformation: {params.Ad}")
    print(f"  Grid size: {params.grid_size}³")
    print("\n🔴 PERTURBATION CONTROLS:")
    print("  Spike - Localized energy spike")
    print("  Wave - Traveling wave")
    print("  Asymmetry - Asymmetric deformation")
    print("  Ring - Ring perturbation")
    print("  Random - Random noise")
    print("  Shock - Shock wave")
    print("  Clear - Remove perturbation")
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050 in your browser")
    print("\nPress Ctrl+C to stop")
    print("="*70)
    
    app.run(debug=True, port=8050)