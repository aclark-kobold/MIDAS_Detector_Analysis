"""
Total number of muons per day hitting an upright hollow cylindrical detector at depth.

Projected area (upright cylinder, top annulus faces up, side wall projects as a 2R x L rectangle):
    A(theta) = pi * (R_out^2 - R_in^2) * cos(theta) + 2 * R_out * L * sin(theta)     (mm^2)

Rate through the detector (azimuth symmetric, downward hemisphere):
    N = 2 pi * integral I(h, theta) * A(theta) * sin(theta) dtheta

I(h, theta) is the underground intensity (/cm^2/s/sr) from muon_diff_energy_spectrum_underground.py,
which propagates the modified Gaisser surface flux through a flat overburden of h m.w.e.
Depths are given in meters of standard rock (2.65 g/cm^3) and converted to m.w.e.
"""
from pathlib import Path
import numpy as np
from muon_diff_energy_spectrum_underground import intensity, _trapz

OUTPUT_FILE = Path(__file__).resolve().parent / "cylinder_muon_rate_vs_depth.txt"

# ----------------------------------------------------------------------------- detector (mm)
R_OUT = 75.0/2
R_IN = 40.1/2
LENGTH = 2500.0

# ----------------------------------------------------------------------------- settings
RHO_ROCK = 2.65            # g/cm^3, standard rock
MM2_TO_CM2 = 1.0e-2
SECONDS_PER_DAY = 86400


def projected_area_cm2(theta):
    """Projected area (cm^2) of the upright cylinder seen from zenith angle theta (rad)."""
    a = np.pi * (R_OUT**2 - R_IN**2) * np.cos(theta) + 2 * R_OUT * LENGTH * np.sin(theta)
    return a * MM2_TO_CM2


def muons_per_second(depth_m, theta_max_deg=89.0, n_theta=181):
    """Muons per second hitting the detector at a vertical rock depth of depth_m meters."""
    h = RHO_ROCK * depth_m  # m.w.e.
    th = np.radians(np.linspace(0.0, theta_max_deg, n_theta))
    I = np.array([intensity(h, t) for t in th])
    return 2 * np.pi * _trapz(I * projected_area_cm2(th) * np.sin(th), th)


def main(depths=np.arange(250, 2001, 250), output_file=OUTPUT_FILE):
    lines = [
        f"Detector: R_out = {R_OUT} mm, R_in = {R_IN} mm, L = {LENGTH} mm (upright)",
        f"Rock density: {RHO_ROCK} g/cm^3",
        "",
        f"{'depth (m)':>10} {'depth (m.w.e.)':>15} {'I_vert (/m^2/s/sr)':>20} {'muons/s':>12} {'muons/day':>12}",
        "-" * 73,
    ]
    for line in lines:
        print(line)
    for d in depths:
        rate = muons_per_second(d)
        I_v = intensity(RHO_ROCK * d, 0.0) * 1.0e4
        line = f"{d:>10.0f} {RHO_ROCK * d:>15.0f} {I_v:>20.3e} {rate:>12.3e} {rate * SECONDS_PER_DAY:>12.1f}"
        print(line)
        lines.append(line)
    Path(output_file).write_text("\n".join(lines) + "\n")
    print(f"\ntable saved to {output_file}")


if __name__ == "__main__":
    main()
