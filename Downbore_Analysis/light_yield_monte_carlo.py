"""
Downbore muon detector: event rate and median light yield versus depth.

One file, numpy only. Three steps:

  1. MUON INTENSITY  I(h, theta): surface flux (modified Gaisser, Guan et al.
     arXiv:1509.06176) carried through rock with a constant energy loss.
     Same model as cylinder_muon_rate_vs_depth.py.
  2. TRACKS  Draw muon tracks from p(theta) ~ I(h, theta) * A_b(theta) * sin(theta),
     throw each one at the tube at a random impact point, and find its path
     (chord) through the scintillator wall.
  3. LIGHT  energy = dE/dx * path; photoelectrons ~ Poisson(energy * k0 * T(z)),
     where k0 is the light per MeV before transport (Table 7 / Appendix D) and
     T(z) is the transport along the fiber from the middle of the track.

The median light yield is the median of the photoelectron count over all tracks.

Run:   python light_yield_monte_carlo.py
Needs: numpy (np.trapezoid in numpy 2.x, np.trapz in older versions).
Time:  about 1 to 2 minutes for 8 depths with 800,000 tracks each.
"""
import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(12345)

# ----------------------------------------------------------------------------
# 0. Inputs (change these to explore; units in the comments)
# ----------------------------------------------------------------------------
R, r, L = 37.5, 20.05, 2500.0      # tube outer radius, bore radius, length (mm)
DEPTHS_M = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]   # depth in metres
RHO_ROCK = 2.65                    # standard rock, g/cm3 -> m.w.e. = metres * 2.65
N_TRACKS = 800_000                 # tracks thrown per depth
DEDX = 2.0                         # MeV/cm, minimum ionization (Appendix E)

# EJ-200 datasheet (Eljen Technology)
Y = 10_000                         # scintillation photons per MeV (1 MeV electron)
LAMBDA_BULK = 3800.0               # bulk light attenuation length, 380 cm (mm)

# Light chain (Table 7). k0 = Y * Pf * eta * f_t * kappa * PDE
ETA = 0.80                         # fiber re-emission efficiency (assumed)
F_T = 0.108                        # trapped in fiber, both directions (2 x 5.4%, Kuraray)
KAPPA = 0.90                       # fiber-to-SiPM coupling (assumed)
PDE = 0.357                        # SiPM PDE at 476 nm, 5.0 V overvoltage (datasheet)
RHO_PAINT = 0.95                   # EJ-510 paint reflectivity
LAMBDA_F = 3.5                     # fiber attenuation length (m, Kuraray)
R_MIRROR = 0.85                    # far-end mirror reflectivity
L_GUIDE = 0.18                     # fiber in the guide (m)
L_TUBE = L / 1000.0                # tube length (m)
L_FIBER = L_GUIDE + L_TUBE         # SiPM end to far-end mirror (m)
N_CELLS = 4774                     # SiPM cells (for saturation)

# ----------------------------------------------------------------------------
# 1. Muon intensity underground: I(h, theta), per cm2 per s per sr
# ----------------------------------------------------------------------------
P1, P2, P3, P4, P5 = 0.102573, -0.068287, 0.958633, 0.0407253, 0.817285   # Guan et al.

def surface_flux(E, theta):
    """Modified Gaisser surface flux, per cm2 s sr GeV. E in GeV, theta in rad."""
    c = np.cos(theta)
    cs = np.sqrt((c**2 + P1**2 + P2 * c**P3 + P4 * c**P5) / (1 + P1**2 + P2 + P4))
    Es = E * (1 + 3.64 / (E * cs**1.29))
    return 0.14 * Es**-2.7 * (1 / (1 + 1.1 * E * cs / 115.0) + 0.054 / (1 + 1.1 * E * cs / 850.0))

A_LOSS, B_LOSS = 0.25, 0.36e-3     # GeV per m.w.e., and 1 per m.w.e. (effective values)
EPS = A_LOSS / B_LOSS              # about 694 GeV

def intensity(h_mwe, theta, e_max=1e9, n=2500):
    """Integrate the surface flux above the minimum surface energy that survives
    X = h/cos(theta) of rock: E_min = EPS * (exp(B X) - 1)."""
    X = h_mwe / np.cos(theta)
    E0 = max(EPS * np.expm1(B_LOSS * X), 0.2)
    if E0 >= e_max:  # no muon within the energy range survives this slant depth
        return 0.0
    E = np.logspace(np.log10(E0), np.log10(e_max), n)
    return trapz(surface_flux(E, theta) * E, np.log(E))

TH = np.radians(np.linspace(0, 88, 441))          # angle grid (rad)

def area_cm2(t):
    """Projected area of the tube (end annulus + side), used for the event RATE."""
    return (np.pi * (R**2 - r**2) * np.cos(t) + 2 * R * L * np.sin(t)) * 1e-2

# ----------------------------------------------------------------------------
# 2. Light chain: fraction absorbed by the fibers, and transport T(z)
# ----------------------------------------------------------------------------
V_MM3 = np.pi * (R**2 - r**2) * L                                   # tube volume (mm3)
S_MM2 = 2*np.pi*R*L + 2*np.pi*r*L + 2*np.pi*(R**2 - r**2)           # surface area (mm2)
ELL_MM = 4 * V_MM3 / S_MM2                                          # mean distance between hits (mm)
A_FIBER = 4 * 0.75 * np.pi * 1.0 * L * 0.90                         # mm2: 4 fibers, 3/4 exposed, 90% absorbed
A_PAINT = (1 - RHO_PAINT) * S_MM2                                   # paint loss area (mm2)
A_BULK = S_MM2 * (1 - np.exp(-ELL_MM / LAMBDA_BULK))                # bulk loss area (mm2)
PF = A_FIBER / (A_FIBER + A_PAINT + A_BULK)                         # fraction absorbed by fibers

def transport(z_m):
    """Fraction of fiber light that reaches the SiPM, for light made at distance
    z (m) from the readout end. Half goes straight to the SiPM, half goes to the
    far-end mirror and comes back."""
    direct = np.exp(-(L_GUIDE + z_m) / LAMBDA_F)
    via_mirror = R_MIRROR * np.exp(-((L_TUBE - z_m) + L_FIBER) / LAMBDA_F)
    return 0.5 * direct + 0.5 * via_mirror

K0 = Y * PF * ETA * F_T * KAPPA * PDE               # pe per MeV before transport

# ----------------------------------------------------------------------------
# 3. Muon tracks through the tube
# ----------------------------------------------------------------------------
def sample_tracks(I_of_theta, n=N_TRACKS):
    """Throw n tracks at the tube; return those that cross the scintillator wall.

    Angle: p(theta) ~ I * sin(theta) * A_b(theta), with A_b the projected area of
    the tube's bounding cylinder, 2R x (L sin + 2R cos). Impact point: uniform
    on that projected rectangle. Tracks that miss the wall are discarded."""
    Ab = 2 * R * (L * np.sin(TH) + 2 * R * np.cos(TH))
    w = I_of_theta * np.sin(TH) * Ab
    cdf = np.cumsum(w); cdf /= cdf[-1]
    theta = np.interp(rng.random(n), cdf, TH)           # inverse-CDF sampling
    phi = rng.random(n) * 2 * np.pi
    d = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)], 1)
    u_ = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], 1)                       # perpendicular to d
    v_ = np.stack([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)], 1)
    uu = (rng.random(n) * 2 - 1) * R
    vv = (rng.random(n) * 2 - 1) * (L * np.sin(theta) / 2 + R * np.cos(theta))
    o = uu[:, None] * u_ + vv[:, None] * v_             # a point on the track

    def chord(rad):
        """Length of the track inside a cylinder of radius rad, cut at z = +/- L/2.
        Also returns the track parameter at the chord midpoint."""
        a = d[:, 0]**2 + d[:, 1]**2
        b = 2 * (o[:, 0]*d[:, 0] + o[:, 1]*d[:, 1])
        c = o[:, 0]**2 + o[:, 1]**2 - rad**2
        disc = b*b - 4*a*c
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        den = np.where(a > 1e-12, 2*a, 1.0)
        t1, t2 = (-b - sq) / den, (-b + sq) / den       # entry and exit of the infinite cylinder
        tz1, tz2 = (-L/2 - o[:, 2]) / d[:, 2], (L/2 - o[:, 2]) / d[:, 2]
        lo = np.maximum(t1, np.minimum(tz1, tz2))       # clip with the two end planes
        hi = np.minimum(t2, np.maximum(tz1, tz2))
        ch = np.where(ok & (hi > lo), hi - lo, 0.0)
        mid = np.where(ch > 0, 0.5 * (lo + hi), 0.0)
        return ch, mid

    ch_out, mid_out = chord(R)
    ch_in, _ = chord(r)
    wall = ch_out - ch_in                               # path in scintillator (mm)
    hit = wall > 1e-6
    z_mid = o[:, 2] + mid_out * d[:, 2]
    return dict(theta=theta[hit], path_mm=wall[hit], z_read_m=((L/2 - z_mid)[hit]) / 1000.0)

def photoelectrons(ev, scale=1.0):
    """Energy deposited (MeV) and photoelectron count for each track."""
    E = DEDX * ev["path_mm"] / 10
    mean_pe = scale * E * K0 * transport(ev["z_read_m"])
    return E, rng.poisson(mean_pe)

def compression(n):
    """SiPM saturation: percent by which n incident pe is under-counted."""
    return 100 * (1 - N_CELLS * (1 - np.exp(-n / N_CELLS)) / n)

# ----------------------------------------------------------------------------
# 4. Results
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("LIGHT BUDGET (Table 7, Table D1)")
    print(f"  Pf = {PF:.3f}  (fiber {A_FIBER/1e6:.4f} m2, paint {A_PAINT/1e6:.4f} m2, bulk {A_BULK/1e6:.4f} m2; "
          f"V = {V_MM3/1e9:.2e} m3, S = {S_MM2/1e6:.3f} m2, mean hop = {ELL_MM:.1f} mm)")
    print(f"  k0 = {K0:.1f} pe/MeV before transport;  transport near/mid/far = "
          f"{transport(0.0):.2f}/{transport(L_TUBE/2):.2f}/{transport(L_TUBE):.2f};  "
          f"yield near/mid/far = {K0*transport(0.0):.1f}/{K0*transport(L_TUBE/2):.1f}/{K0*transport(L_TUBE):.1f} pe/MeV\n")

    print("EVENT RATE (user's model; area = end annulus + side)")
    print(" depth   mean zenith   events/day   per 30 d   precision (1/sqrt(N))")
    for dm in DEPTHS_M:
        I = np.array([intensity(dm * RHO_ROCK, t) for t in TH])
        w = I * area_cm2(TH) * np.sin(TH)
        per_day = 2 * np.pi * trapz(w, TH) * 86400
        mean_zen = np.degrees(trapz(w * TH, TH) / trapz(w, TH))
        print(f" {dm:5d}   {mean_zen:7.1f} deg   {per_day:9.2f}   {per_day*30:8.0f}   {100/np.sqrt(per_day*30):7.1f}%")

    print("\nMEDIAN LIGHT YIELD (Monte Carlo, all tracks that cross the wall)")
    print(" depth | path med (mm) | E med (MeV) [16-84%]  | pe med [16-84%]    | half / double | >=20pe: nom / half | compression: median / 84th / 1000pe")
    for dm in DEPTHS_M:
        I = np.array([intensity(dm * RHO_ROCK, t) for t in TH])
        ev = sample_tracks(I)
        E, pe = photoelectrons(ev)
        _, pe_half = photoelectrons(ev, 0.5); _, pe_dbl = photoelectrons(ev, 2.0)
        qe = np.percentile(E, [16, 50, 84]); qp = np.percentile(pe, [16, 50, 84])
        print(f" {dm:5d} | {np.median(ev['path_mm']):8.1f}      | {qe[1]:5.2f} [{qe[0]:4.1f}-{qe[2]:4.1f}]      | "
              f"{qp[1]:4.0f} [{qp[0]:3.0f}-{qp[2]:3.0f}]    | {np.median(pe_half):4.0f} / {np.median(pe_dbl):4.0f} | "
              f"{100*(pe >= 20).mean():5.2f}% / {100*(pe_half >= 20).mean():5.2f}%    | "
              f"{compression(qp[1]):4.1f}% / {compression(qp[2]):4.1f}% / {compression(1000):4.1f}%")

    print("\nSURFACE CHECK (threshold that keeps 95% of crossing muons, surface angular distribution)")
    I0 = np.array([intensity(0.0, t) for t in TH])
    ev = sample_tracks(I0); _, pe = photoelectrons(ev)
    print(f"  5th percentile of the photoelectron count at the surface = {np.percentile(pe, 5):.0f} pe")
