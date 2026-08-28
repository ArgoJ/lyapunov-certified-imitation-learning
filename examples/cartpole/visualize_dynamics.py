"""Interactive closed-loop Acados MPC simulation and physical animation for Cartpole."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch as th

from examples.cartpole.acados_ocp import get_ocp_solver
from examples.cartpole.basis import Q, R
from examples.cartpole.cartpole_dyn import CartpoleDynamics
from examples.cartpole.sys_cfg import PendulumOnCartConfig
from lcil.utils import IntegrationMethod


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for initial state and simulation parameters."""
    parser = argparse.ArgumentParser(
        description="Simulate and visualize Cartpole closed-loop Acados MPC with PyTorch dynamics."
    )
    parser.add_argument(
        "--x0",
        type=float,
        nargs=4,
        default=None,
        metavar=("P", "V", "THETA", "THETA_DOT"),
        help="Full initial state vector: [cart_pos (m), cart_vel (m/s), pole_angle (rad), pole_ang_vel (rad/s)].",
    )
    parser.add_argument(
        "--p",
        "--cart-pos",
        dest="cart_pos",
        type=float,
        default=0.0,
        help="Initial cart position p0 [m] (default: 0.0).",
    )
    parser.add_argument(
        "--v",
        "--cart-vel",
        dest="cart_vel",
        type=float,
        default=0.0,
        help="Initial cart velocity v0 [m/s] (default: 0.0).",
    )
    parser.add_argument(
        "--theta",
        "--pole-angle",
        dest="pole_angle",
        type=float,
        default=0.25,
        help="Initial pole angle theta0 [rad] (default: 0.25 rad ≈ 14.3°).",
    )
    parser.add_argument(
        "--theta-deg",
        dest="pole_angle_deg",
        type=float,
        default=None,
        help="Initial pole angle theta0 in degrees (overrides --theta).",
    )
    parser.add_argument(
        "--theta-dot",
        "--pole-ang-vel",
        dest="pole_ang_vel",
        type=float,
        default=0.0,
        help="Initial pole angular velocity dtheta0 [rad/s] (default: 0.0).",
    )
    parser.add_argument(
        "--t-sim",
        "--steps",
        dest="t_sim",
        type=int,
        default=100,
        help="Number of simulation steps (default: 100).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Discretization time step dt [s] (default: 0.05).",
    )
    parser.add_argument(
        "--N",
        dest="n_horizon",
        type=int,
        default=40,
        help="MPC prediction horizon N (default: 40).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/cartpole",
        help="Output directory for generated HTML files (default: examples/cartpole).",
    )

    return parser.parse_args()


def simulate_closed_loop_mpc(
    solver,
    dyn_pt: CartpoleDynamics,
    x0: np.ndarray,
    T_sim: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Simulate closed-loop stabilization with Acados MPC controlling PyTorch Cartpole dynamics."""
    states = [x0.copy()]
    controls = []
    statuses = []

    x_curr = x0.copy()
    x_curr_pt = th.tensor(x0, dtype=th.float32).unsqueeze(0)

    for t in range(T_sim):
        # Set state feedback in Acados OCP solver
        solver.set(0, "lbx", x_curr)
        solver.set(0, "ubx", x_curr)

        status = solver.solve()
        statuses.append(status)
        u_opt = np.array(solver.get(0, "u")).flatten()
        u_val = float(u_opt[0])
        controls.append(u_val)

        # Propagate PyTorch model
        u_pt = th.tensor([[u_val]], dtype=th.float32)
        with th.no_grad():
            x_next_pt = dyn_pt(x_curr_pt, u_pt)

        x_next_np = x_next_pt.squeeze(0).numpy().copy()
        states.append(x_next_np)
        x_curr = x_next_np
        x_curr_pt = x_next_pt

    return np.array(states), np.array(controls), statuses


def build_mpc_dashboard(
    time_vec: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    statuses: list[int],
    x0: np.ndarray,
    dt: float,
) -> go.Figure:
    """Build a 6-panel Plotly dashboard showing state trajectories, control force, and phase portrait."""
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "<b>Cart Position p(t) [m]</b> (Target: 0.0 m)",
            "<b>Pendulum Angle θ(t) [rad]</b> (Target: 0.0 rad - Upright)",
            "<b>Cart Velocity v(t) [m/s]</b>",
            "<b>Angular Velocity dθ/dt [rad/s]</b>",
            "<b>MPC Optimal Force F(t) [N]</b>",
            "<b>Phase Portrait (θ vs dθ/dt)</b>",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    color_state = "#1f77b4"  # Blue
    color_force = "#2ca02c"  # Green
    color_phase = "#e74c3c"  # Red

    # 1. Cart Position
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=states[:, 0],
            name="Cart Position p",
            line=dict(color=color_state, width=2.5),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Pendulum Angle
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=states[:, 2],
            name="Pole Angle θ",
            line=dict(color=color_state, width=2.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Cart Velocity
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=states[:, 1],
            name="Cart Velocity v",
            line=dict(color=color_state, width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Angular Velocity
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=states[:, 3],
            name="Angular Velocity dθ",
            line=dict(color=color_state, width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 5. Control Force
    fig.add_trace(
        go.Scatter(
            x=time_vec[:-1],
            y=controls,
            name="MPC Force F",
            line=dict(color=color_force, width=2),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # 6. Phase Portrait (theta vs dtheta)
    fig.add_trace(
        go.Scatter(
            x=states[:, 2],
            y=states[:, 3],
            mode="lines+markers",
            name="Phase Trajectory",
            line=dict(color=color_phase, width=2),
            marker=dict(size=4, color=color_phase),
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    # Highlight start and target in phase portrait
    fig.add_trace(
        go.Scatter(
            x=[states[0, 2]],
            y=[states[0, 3]],
            mode="markers",
            marker=dict(size=10, color="#f39c12", symbol="circle"),
            name="Start (t=0)",
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            marker=dict(size=12, color="#27ae60", symbol="star"),
            name="Equilibrium (0, 0)",
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    # Zero line references
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]:
        fig.add_hline(y=0.0, line=dict(color="#bbb", width=1, dash="dot"), row=r, col=c)
    fig.add_hline(y=0.0, line=dict(color="#bbb", width=1, dash="dot"), row=3, col=2)
    fig.add_vline(x=0.0, line=dict(color="#bbb", width=1, dash="dot"), row=3, col=2)

    fig.update_xaxes(title_text="Time t [s]", row=1, col=1)
    fig.update_xaxes(title_text="Time t [s]", row=1, col=2)
    fig.update_xaxes(title_text="Time t [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time t [s]", row=2, col=2)
    fig.update_xaxes(title_text="Time t [s]", row=3, col=1)
    fig.update_xaxes(title_text="Angle θ [rad]", row=3, col=2)
    fig.update_yaxes(title_text="Angular Vel dθ/dt [rad/s]", row=3, col=2)

    status_ok = all(s == 0 for s in statuses)
    status_badge = (
        "<b style='color:#27ae60;'>[✓ ALL MPC SOLVES OPTIMAL]</b>"
        if status_ok
        else "<b style='color:#e74c3c;'>[⚠ SOME SOLVES INFEASIBLE]</b>"
    )
    deg0 = x0[2] * 180.0 / np.pi

    title_text = (
        "<b>Acados Closed-Loop MPC Cartpole Stabilization</b><br>"
        f"<span style='font-size: 13px; color: #444;'>"
        f"Initial State: p₀ = {x0[0]:.2f} m, v₀ = {x0[1]:.2f} m/s, θ₀ = {x0[2]:.3f} rad ({deg0:.1f}°), dθ₀ = {x0[3]:.2f} rad/s "
        f"| dt = {dt}s | {status_badge}</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.98, xanchor="center", yanchor="top"),
        margin=dict(t=95, b=50, l=60, r=40),
        template="plotly_white",
        height=850,
        width=1100,
    )

    return fig


def build_physical_animation_figure(
    time_vec: np.ndarray,
    states: np.ndarray,
    sys_cfg: PendulumOnCartConfig,
    downsample: int = 1,
) -> go.Figure:
    """Construct a clean 2D physical animation of the cart and inverted pendulum without vertical offsets."""
    l = sys_cfg.length
    cart_w = 0.35
    cart_h = 0.18

    idx = np.arange(0, len(time_vec), downsample)
    t_frames = time_vec[idx]
    p_frames = states[idx, 0]
    theta_frames = states[idx, 2]

    # Tip position relative to cart pivot at (p, 0)
    tip_x_frames = p_frames + l * np.sin(theta_frames)
    tip_y_frames = l * np.cos(theta_frames)

    x_min = min(np.min(p_frames), np.min(tip_x_frames), -1.0) - 0.4
    x_max = max(np.max(p_frames), np.max(tip_x_frames), 1.0) + 0.4
    y_min = -l - 0.15
    y_max = l + 0.25

    p0 = p_frames[0]
    tx0 = tip_x_frames[0]
    ty0 = tip_y_frames[0]

    fig = go.Figure(
        data=[
            # 1. Ground rail line exactly at bottom of cart y = -cart_h/2
            go.Scatter(
                x=[x_min, x_max],
                y=[-cart_h / 2, -cart_h / 2],
                mode="lines",
                line=dict(color="#555", width=3),
                name="Ground Rail",
                hoverinfo="skip",
            ),
            # 2. Target Origin Marker at (0, l)
            go.Scatter(
                x=[0.0],
                y=[l],
                mode="markers",
                marker=dict(size=14, symbol="cross", color="#27ae60", line=dict(width=2.5, color="#1e8449")),
                name="Target Upright Equilibrium (0, l)",
            ),
            # 3. Cart Box centered at y = 0
            go.Scatter(
                x=[p0 - cart_w / 2, p0 + cart_w / 2, p0 + cart_w / 2, p0 - cart_w / 2, p0 - cart_w / 2],
                y=[-cart_h / 2, -cart_h / 2, cart_h / 2, cart_h / 2, -cart_h / 2],
                fill="toself",
                fillcolor="#3498db",
                mode="lines",
                line=dict(color="#2980b9", width=2.5),
                name="Cart",
            ),
            # 4. Pendulum Rod & Tip Mass pivoting at (p, 0)
            go.Scatter(
                x=[p0, tx0],
                y=[0.0, ty0],
                mode="lines+markers",
                line=dict(color="#e74c3c", width=5),
                marker=dict(size=[8, 20], color=["#2c3e50", "#c0392b"]),
                name="Pendulum Rod & Tip Mass",
            ),
        ],
        layout=go.Layout(
            title=dict(
                text="<b>2D Physical Cartpole Animation (Acados Closed-Loop MPC)</b><br>"
                     "<span style='font-size: 13px; color: #555;'>Click ▶ Play to watch real-time balancing</span>",
                x=0.5,
                y=0.97,
                xanchor="center",
                yanchor="top",
            ),
            xaxis=dict(
                range=[x_min, x_max],
                title="<b>Horizontal Position x [m]</b>",
                zeroline=True,
                zerolinecolor="#ddd",
            ),
            yaxis=dict(
                range=[y_min, y_max],
                title="<b>Vertical Position y [m]</b>",
                scaleanchor="x",
                scaleratio=1,
                zeroline=True,
                zerolinecolor="#eee",
            ),
            margin=dict(t=90, b=40, l=60, r=40),
            template="plotly_white",
            height=580,
            width=960,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.10,
                    x=0.0,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=45, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=13),
                        prefix="Time: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=25),
                    len=0.9,
                    x=0.05,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [f"frame_{k}"],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{t_frames[k]:.2f}s",
                            method="animate",
                        )
                        for k in range(len(t_frames))
                    ],
                )
            ],
        ),
    )

    # Build animation frames
    frames = []
    for k in range(len(t_frames)):
        pk = p_frames[k]
        txk = tip_x_frames[k]
        tyk = tip_y_frames[k]

        frame_data = [
            # Rail
            go.Scatter(x=[x_min, x_max], y=[-cart_h / 2, -cart_h / 2]),
            # Target
            go.Scatter(x=[0.0], y=[l]),
            # Cart (centered at y = 0)
            go.Scatter(
                x=[pk - cart_w / 2, pk + cart_w / 2, pk + cart_w / 2, pk - cart_w / 2, pk - cart_w / 2],
                y=[-cart_h / 2, -cart_h / 2, cart_h / 2, cart_h / 2, -cart_h / 2],
            ),
            # Pendulum (pivots at (pk, 0))
            go.Scatter(
                x=[pk, txk],
                y=[0.0, tyk],
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame_{k}"))

    fig.frames = frames
    return fig


def main() -> None:
    args = parse_args()

    # Determine initial state x0
    if args.x0 is not None:
        x0 = np.array(args.x0, dtype=np.float64)
    else:
        p0 = args.cart_pos
        v0 = args.cart_vel
        if args.pole_angle_deg is not None:
            theta0 = args.pole_angle_deg * np.pi / 180.0
        else:
            theta0 = args.pole_angle
        dtheta0 = args.pole_ang_vel
        x0 = np.array([p0, v0, theta0, dtheta0], dtype=np.float64)

    sys_cfg = PendulumOnCartConfig()
    dt = args.dt
    N_horizon = args.n_horizon
    T_sim = args.t_sim
    time_vec = np.arange(T_sim + 1) * dt

    print(f"=== Cartpole Acados MPC Simulation (PyTorch Dynamics) ===")
    print(f"Initial state x0 = [p: {x0[0]:.2f}m, v: {x0[1]:.2f}m/s, θ: {x0[2]:.3f}rad ({x0[2]*180/np.pi:.1f}°), dθ: {x0[3]:.2f}rad/s]")
    print(f"Horizon N = {N_horizon}, dt = {dt}s, Total steps = {T_sim} ({T_sim * dt:.1f}s)")

    print("\nInitializing Acados OCP solver...")
    solver, info = get_ocp_solver(
        Q=Q,
        R=R,
        dt=dt,
        N=N_horizon,
        terminal_mode="regional",
        sys_cfg=sys_cfg,
        use_temp_dir=True,
    )

    pt_dyn = CartpoleDynamics(
        dt=dt,
        sys_cfg=sys_cfg,
        method=IntegrationMethod.EXPLICIT_EULER,
    )

    print("Simulating closed-loop Acados MPC on PyTorch dynamics...")
    states, controls, statuses = simulate_closed_loop_mpc(
        solver=solver,
        dyn_pt=pt_dyn,
        x0=x0,
        T_sim=T_sim,
    )

    print(f"\nSimulation completed. All MPC solves optimal: {all(s == 0 for s in statuses)}")

    out_path = Path(args.output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. State trajectory & control dashboard
    dashboard_fig = build_mpc_dashboard(
        time_vec=time_vec,
        states=states,
        controls=controls,
        statuses=statuses,
        x0=x0,
        dt=dt,
    )
    dashboard_file = out_path / "cartpole_mpc_closed_loop_verification.html"
    dashboard_fig.write_html(str(dashboard_file))
    print(f"[✓] Saved MPC Dashboard to: {dashboard_file}")

    # 2. 2D Physical Animation Figure
    anim_fig = build_physical_animation_figure(
        time_vec=time_vec,
        states=states,
        sys_cfg=sys_cfg,
    )
    anim_file = out_path / "cartpole_mpc_physical_animation.html"
    anim_fig.write_html(str(anim_file))
    print(f"[✓] Saved 2D Physical Animation to: {anim_file}")


if __name__ == "__main__":
    main()
