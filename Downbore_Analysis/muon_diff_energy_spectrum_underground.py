"""
Underground muon spectra and angular distributions from a surface muon model (modified Gaisser).

Method (flat overburden, continuous energy loss  dE/dX = -a - bE):
    slant depth            X = h / cos(theta)                         (h in m.w.e.)
    epsilon                = a / b
    minimum surface energy E_min = eps * (exp(bX) - 1)
    surface energy         E_s = (E_u + eps) * exp(bX) - eps          (E_u = energy underground)
    spectrum at depth      Phi_u(E_u, h, theta) = exp(bX) * Phi_s(E_s, theta)
    intensity              I(h, theta) = integral of Phi_s(E, theta) dE   from E_min to infinity
    mean energy            <E_u> = integral E_u Phi_u dE_u / I
    total flux (MUTE def.) Phi_tot = 2 pi * integral I(h, theta) sin(theta) dtheta
    flux through a flat horizontal surface = 2 pi * integral I cos(theta) sin(theta) dtheta

Units: E in GeV; fluxes in /cm^2/s/sr/GeV internally.  Plots are in /m^2 if PER_M2 is True.

Loss parameters: A_LOSS = 0.25 GeV/m.w.e. and B_LOSS = 0.36e-3 per m.w.e. are EFFECTIVE values that
reproduce the published MUTE + daemonflux total fluxes within about 4% at 2.0, 2.8 and 2.9 km.w.e.
(standard-rock-like).  Change them for your rock.  Reliable roughly for 0.5 to 6 km.w.e.

Surface model: `modified_gaisser_formula(E, theta)` from ../Muon_Surface_Models/guan_model.py
(E in GeV, theta in radians, 0 to 90 deg), converted from /m^2 to /cm^2/s/sr/GeV.

Run:  python muon_diff_energy_spectrum_underground.py       (saves figures in ./Plots next to this file)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Make the surface models in ../Muon_Surface_Models importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Muon_Surface_Models"))

from guan_model import modified_gaisser_formula

# ----------------------------------------------------------------------------- surface flux model
def surface_flux(E, theta):
    """Modified Gaisser (Guan) surface flux: /cm^2/s/sr/GeV."""
    return modified_gaisser_formula(E, theta) / 10000

# ----------------------------------------------------------------------------- settings
A_LOSS = 0.25            # GeV per m.w.e.   (ionization loss)
B_LOSS = 0.36e-3         # per m.w.e.       (radiative loss)
EPS = A_LOSS / B_LOSS    # GeV
PER_M2 = True            # plots in /m^2 (True) or /cm^2 (False)
UNIT = 1.0e4 if PER_M2 else 1.0
AREA = "m" if PER_M2 else "cm"


def _trapz(y, x, axis=-1):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=axis)


# ----------------------------------------------------------------------------- core physics
def slant_depth(h, theta):
    """Slant depth (m.w.e.) for a flat overburden of vertical depth h and zenith angle theta (rad)."""
    return h / np.cos(theta)


def e_min(X):
    """Minimum surface energy (GeV) for a muon to reach slant depth X (m.w.e.)."""
    return EPS * np.expm1(B_LOSS * X)


def surface_energy(E_u, X):
    """Surface energy (GeV) of a muon that arrives with energy E_u after slant depth X."""
    return (E_u + EPS) * np.exp(B_LOSS * X) - EPS


def underground_spectrum(E_u, h, theta):
    """dN/dE_u at depth h, zenith angle theta: /cm^2/s/sr/GeV  (E_u may be an array)."""
    X = slant_depth(h, theta)
    return np.exp(B_LOSS * X) * surface_flux(surface_energy(E_u, X), theta)


def intensity(h, theta, e_max=1.0e9, n=2500):
    """Underground intensity /cm^2/s/sr: surface flux integrated above E_min (integrated in ln E)."""
    X = slant_depth(h, theta)
    E0 = max(e_min(X), 0.2)
    if E0 >= e_max:  # no muon within the energy range survives this slant depth
        return 0.0
    E = np.logspace(np.log10(E0), np.log10(e_max), n)
    return _trapz(surface_flux(E, theta) * E, np.log(E))


def mean_energy(h, theta, n=3000):
    """Mean underground muon energy (GeV) at depth h and zenith angle theta."""
    E = np.logspace(-2, 6, n)
    f = underground_spectrum(E, h, theta)
    return _trapz(f * E * E, np.log(E)) / _trapz(f * E, np.log(E))


def spectrum_all_angles(E_u, h, theta_max_deg=80.0, n_theta=41):
    """Angle-integrated spectrum, 2 pi * integral Phi_u sin(theta) dtheta: /cm^2/s/GeV."""
    th = np.radians(np.linspace(0.0, theta_max_deg, n_theta))
    vals = np.array([underground_spectrum(E_u, h, t) * np.sin(t) for t in th])
    return 2 * np.pi * _trapz(vals, th, axis=0)


def total_flux(h, theta_max_deg=80.0, n_theta=81, horizontal=False):
    """Total flux /cm^2/s.  horizontal=False: all directions (MUTE definition);
    horizontal=True: flux crossing a flat horizontal surface (cos(theta) weighted)."""
    th = np.radians(np.linspace(0.0, theta_max_deg, n_theta))
    w = np.sin(th) * (np.cos(th) if horizontal else 1.0)
    I = np.array([intensity(h, t) for t in th])
    return 2 * np.pi * _trapz(I * w, th)


def mean_energy_all_angles(h, theta_max_deg=80.0, n_theta=41, n_E=1500):
    E = np.logspace(-2, 6, n_E)
    f = spectrum_all_angles(E, h, theta_max_deg, n_theta)
    return _trapz(f * E * E, np.log(E)) / _trapz(f * E, np.log(E))


def effective_n(h, theta_deg):
    """Effective cos^n index: I(theta)/I(0) = cos(theta)^n."""
    th = np.radians(theta_deg)
    return np.log(intensity(h, th) / intensity(h, 0.0)) / np.log(np.cos(th))


def arrival_angle_pdf(h, theta_max_deg=80.0, n_theta=81, horizontal=False):
    """Normalized zenith-angle distribution dN/dtheta (per deg) at depth h.
    horizontal=False: all directions, I sin(theta); horizontal=True: through a flat surface, I sin cos."""
    thd = np.linspace(0.0, theta_max_deg, n_theta)
    th = np.radians(thd)
    w = np.sin(th) * (np.cos(th) if horizontal else 1.0)
    p = np.array([intensity(h, t) for t in th]) * w
    return thd, p / _trapz(p, thd)


def arrival_angle_stats(h, horizontal=False, n_theta=81):
    """Mean, median, 16th and 84th percentile of the arrival zenith angle (deg) at depth h."""
    thd, p = arrival_angle_pdf(h, n_theta=n_theta, horizontal=horizontal)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(thd))])
    cdf /= cdf[-1]
    mean = _trapz(thd * p, thd)
    p16, med, p84 = np.interp([0.16, 0.5, 0.84], cdf, thd)
    return mean, med, p16, p84


# ----------------------------------------------------------------------------- checks
MUTE_REF = [(2929, 3.58e-8, "SUPL (standard rock)"), (2803, 4.19e-8, "Boulby"), (2032, 1.66e-7, "Soudan")]


def self_check():
    print("surface model: guan_model.modified_gaisser_formula")
    print(f"loss parameters: a = {A_LOSS} GeV/m.w.e., b = {B_LOSS:.2e} /m.w.e., eps = {EPS:.0f} GeV")
    # 1) the spectrum at depth must integrate to the intensity
    for h in (1000, 2650):
        E = np.logspace(-2, 6, 4000)
        a = _trapz(underground_spectrum(E, h, 0.0) * E, np.log(E))
        b = intensity(h, 0.0)
        print(f"  h={h}: integral of Phi_u = {a:.4e}   intensity = {b:.4e}   (should agree)")
    # 2) total flux versus published MUTE + daemonflux values (all directions, /cm^2/s)
    print("  total flux (all directions) versus MUTE + daemonflux:")
    for h, ref, site in MUTE_REF:
        t = total_flux(h)
        print(f"    h={h}: this code {t:.2e}   MUTE {ref:.2e}   ratio {t / ref:.2f}   ({site})")
    print(f"  vertical intensity at 2650 m.w.e.: {intensity(2650, 0.0):.2e} /cm^2/s/sr")


# ----------------------------------------------------------------------------- plots
def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)


def plot_vertical_spectra(depths, outdir):
    E = np.logspace(0, 5, 400)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].loglog(E, surface_flux(E, 0.0) * UNIT, "k--", lw=1.3, label="surface")
    ax[1].loglog(E, E * surface_flux(E, 0.0) * UNIT, "k--", lw=1.3, label="surface")
    for h in depths:
        f = underground_spectrum(E, h, 0.0) * UNIT
        ax[0].loglog(E, f, label=f"{h / 1000:g} km.w.e.")
        ax[1].loglog(E, E * f, label=f"{h / 1000:g} km.w.e.")
    _style(ax[0], "Muon energy (GeV)", f"dN/dE  (1/{AREA}$^2$/s/sr/GeV)", "Vertical differential spectrum")
    _style(ax[1], "Muon energy (GeV)", f"E x dN/dE  (1/{AREA}$^2$/s/sr)", "Vertical energy-weighted spectrum")
    ax[0].set_ylim(bottom=1e-14 * UNIT / 1e4)
    ax[1].set_ylim(bottom=1e-12 * UNIT / 1e4)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "spectra_vertical.png", dpi=200, bbox_inches="tight")
    return fig


def plot_spectra_all_angles(depths, outdir):
    E = np.logspace(0, 5, 300)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    fs = spectrum_all_angles(E, 0.0) * UNIT
    ax[0].loglog(E, fs, "k--", lw=1.3, label="surface")
    ax[1].loglog(E, E * fs, "k--", lw=1.3, label="surface")
    for h in depths:
        f = spectrum_all_angles(E, h) * UNIT
        ax[0].loglog(E, f, label=f"{h / 1000:g} km.w.e.")
        ax[1].loglog(E, E * f, label=f"{h / 1000:g} km.w.e.")
    _style(ax[0], "Muon energy at depth (GeV)", f"dN/dE  (1/{AREA}$^2$/s/GeV)", "Angle-integrated differential spectrum")
    _style(ax[1], "Muon energy at depth (GeV)", f"E x dN/dE  (1/{AREA}$^2$/s)", "Angle-integrated energy-weighted spectrum")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "spectra_all_angles.png", dpi=200, bbox_inches="tight")
    return fig


def plot_depth_intensity(outdir):
    hs = np.linspace(500, 6000, 30)
    Iv = np.array([intensity(h, 0.0) for h in hs]) * UNIT
    Tot = np.array([total_flux(h, n_theta=41) for h in hs]) * UNIT
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(hs / 1000, Iv, label=f"vertical intensity (1/{AREA}$^2$/s/sr)")
    ax.semilogy(hs / 1000, Tot, label=f"total flux, all directions (1/{AREA}$^2$/s)")
    ref_h = [r[0] / 1000 for r in MUTE_REF]
    ref_v = [r[1] * UNIT for r in MUTE_REF]
    ax.semilogy(ref_h, ref_v, "ko", label="MUTE + daemonflux totals")
    _style(ax, "Vertical depth (km.w.e.)", "vertical intensity or total flux (units in legend)", "Depth-intensity relation (flat standard rock)")
    ax.set_xlim(0, 6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "depth_intensity.png", dpi=200, bbox_inches="tight")
    return fig


def plot_angular(depths, outdir):
    thd = np.linspace(0, 80, 41)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(thd, np.cos(np.radians(thd)) ** 2, "k--", lw=1.2, label="cos$^2\\theta$ (surface, ~GeV)")
    for h in depths:
        I0 = intensity(h, 0.0)
        ratio = np.array([intensity(h, np.radians(t)) for t in thd]) / I0
        ax[0].semilogy(thd, ratio, label=f"{h / 1000:g} km.w.e.")
        nn = np.array([np.nan if t == 0 else np.log(r) / np.log(np.cos(np.radians(t))) for t, r in zip(thd, ratio)])
        ax[1].plot(thd, nn, label=f"{h / 1000:g} km.w.e.")
    _style(ax[0], "Zenith angle (deg)", "I($\\theta$) / I(0)", "Angular distribution at depth (per sr)")
    ax[0].set_ylim(1e-4, 1.5)
    _style(ax[1], "Zenith angle (deg)", "effective index n in cos$^n\\theta$", "Effective cosine power")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "angular_distribution.png", dpi=200, bbox_inches="tight")
    return fig


def plot_arrival_angle_vs_depth(depths, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for h in depths:
        thd, p = arrival_angle_pdf(h)
        ax[0].plot(thd, p, label=f"{h / 1000:g} km.w.e.")
    _style(ax[0], "Zenith angle (deg)", "dN/d$\\theta$  (normalized, 1/deg)", "Arrival zenith-angle distribution (all directions)")
    ax[0].legend(fontsize=8)

    hs = np.linspace(500, 6000, 23)
    s = np.array([arrival_angle_stats(h) for h in hs])
    ax[1].plot(hs / 1000, s[:, 0], color="C0", label="mean")
    ax[1].plot(hs / 1000, s[:, 1], "--", color="C0", label="median")
    ax[1].fill_between(hs / 1000, s[:, 2], s[:, 3], color="C0", alpha=0.15, label="16-84%")
    _style(ax[1], "Vertical depth (km.w.e.)", "Arrival zenith angle (deg)", "Muon arrival angle versus depth (all directions)")
    ax[1].set_xlim(0, 6)
    ax[1].set_ylim(bottom=0)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "arrival_angle_vs_depth.png", dpi=200, bbox_inches="tight")
    return fig


def plot_mean_energy(outdir):
    hs = np.linspace(500, 6000, 23)
    mv = [mean_energy(h, 0.0) for h in hs]
    ma = [mean_energy_all_angles(h) for h in hs]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(hs / 1000, mv, label="vertical")
    ax.plot(hs / 1000, ma, label="all directions")
    _style(ax, "Vertical depth (km.w.e.)", "mean muon energy underground (GeV)", "Mean muon energy versus depth")
    ax.set_xlim(0, 6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mean_energy.png", dpi=200, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- main
def main(depths=(500, 1000, 1500, 2000, 3000, 4000, 6000), outdir=HERE / "Plots", show=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    self_check()
    if min(depths) < 400:
        print("WARNING: below ~0.4 km.w.e. the minimum surface energy drops under ~100 GeV and the "
              "constant-a,b energy-loss model becomes unreliable.")
    plot_vertical_spectra(depths, outdir)
    plot_spectra_all_angles(depths, outdir)
    plot_depth_intensity(outdir)
    plot_angular(depths, outdir)
    plot_arrival_angle_vs_depth(depths, outdir)
    plot_mean_energy(outdir)
    print(f"figures saved in {outdir.resolve()}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()