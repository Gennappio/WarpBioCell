"""Run figures with matplotlib (optional dependency). Nothing here touches the simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from warpbiocell.cells.model import CellState  # noqa: E402

STATE_COLORS = {
    CellState.PROLIFERATIVE: "#2a9d8f",
    CellState.QUIESCENT: "#e9c46a",
    CellState.HYPOXIC: "#f4a261",
    CellState.DEAD: "#6c757d",
}


def population_figure(result, path: Path) -> Path:
    t = result.metric("time_h") / 24.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, result.metric("living_cells"), "k-", label="living")
    for state in CellState:
        ax1.plot(t, result.metric(f"{state.name.lower()}_cells"), color=STATE_COLORS[state], label=state.name.lower())
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel("cells")
    ax1.legend(frameon=False)
    ax1.set_title("population by state")

    ax2.plot(t, result.metric("spheroid_radius_um"), "k-", label="spheroid (R99)")
    ax2.plot(t, result.metric("non_proliferative_radius_um"), color=STATE_COLORS[CellState.QUIESCENT], label="non-proliferative core")
    ax2.plot(t, result.metric("hypoxic_radius_um"), color=STATE_COLORS[CellState.HYPOXIC], label="hypoxic core")
    ax2.plot(t, result.metric("necrotic_radius_um"), color=STATE_COLORS[CellState.DEAD], label="necrotic core")
    ax2.set_xlabel("time [days]")
    ax2.set_ylabel("radius [um]")
    ax2.legend(frameon=False)
    ax2.set_title("characteristic radii")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def radial_profile_figure(time_h: float, profile: dict, path: Path) -> Path:
    r = profile["r_outer"]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    for state in CellState:
        ax1.plot(r, profile[f"fraction_{state.name.lower()}"], "o-", ms=3, color=STATE_COLORS[state], label=state.name.lower())
    ax1.set_xlabel("shell outer radius [um]")
    ax1.set_ylabel("fraction of cells")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(r, profile["oxygen_mean"], "k--", label="oxygen")
    ax2.set_ylabel("oxygen [mmHg]")
    ax2.set_ylim(bottom=0)
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=8)
    ax1.set_title(f"radial profile at t = {time_h / 24:.1f} days")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def oxygen_slice_figure(population, oxygen, time_h: float, path: Path) -> Path:
    geom = oxygen.geometry
    field = oxygen.numpy()
    k = geom.shape[2] // 2
    z0 = geom.origin[2] + k * geom.dx
    x_axis, y_axis, _ = geom.node_coordinates()
    pos = population.positions_numpy()
    states = population.states_numpy()
    radii = population.radii_numpy()
    in_slab = np.abs(pos[:, 2] - z0) < radii

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(
        field[:, :, k].T,
        origin="lower",
        extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
        cmap="Blues",
        vmin=0.0,
        vmax=max(float(field.max()), 1e-6),
    )
    fig.colorbar(im, ax=ax, label="oxygen [mmHg]")
    for state in CellState:
        sel = in_slab & (states == int(state))
        if sel.any():
            ax.scatter(pos[sel, 0], pos[sel, 1], s=6, color=STATE_COLORS[state], label=state.name.lower(), linewidths=0)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"mid-plane oxygen and cells at t = {time_h / 24:.1f} days")
    ax.legend(frameon=False, loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_run_figures(directory: Path, result, population, oxygen) -> list[Path]:
    figures = Path(directory) / "figures"
    figures.mkdir(exist_ok=True)
    made = [population_figure(result, figures / "population.png")]
    if result.profiles:
        time_h, profile = result.profiles[-1]
        made.append(radial_profile_figure(time_h, profile, figures / "radial_profile.png"))
    if oxygen is not None:
        made.append(oxygen_slice_figure(population, oxygen, result.metrics[-1]["time_h"], figures / "oxygen_slice.png"))
    return made
