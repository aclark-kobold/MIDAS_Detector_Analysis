# MIDAS Detector Analysis

Estimates of the cosmic-ray muon rates a MIDAS detector would see, both at the surface and underground in a borehole (downbore).

All rates start from one analytical surface muon flux model. For the downbore case, that flux is propagated through a flat rock overburden and then integrated over the detector's projected area.

## Layout

```
MIDAS_Detector_Analysis/
├── Muon_Surface_Models/
│   └── guan_model.py                          surface muon flux model (shared)
├── Surface_Analysis/                          surface detector rates (to be added)
└── Downbore_Analysis/
    ├── muon_diff_energy_spectrum_underground.py   surface flux -> underground spectra and intensity
    ├── cylinder_muon_rate_vs_depth.py             muons/day on the downbore tube vs depth
    ├── light_yield_monte_carlo.py                 event rate + photoelectron yield Monte Carlo
    ├── cylinder_muon_rate_vs_depth.txt            output table
    └── Plots/                                     output figures
```

## Requirements

- Python 3
- numpy (1.x or 2.x)
- matplotlib (only for `muon_diff_energy_spectrum_underground.py`)

## Physics summary

### Surface flux: `Muon_Surface_Models/guan_model.py`

`modified_gaisser_formula(E, theta)` is the modified Gaisser parameterization of Guan et al. (arXiv:1509.06176). It is valid down to low energies and at all zenith angles.

- E in GeV, theta (zenith) in radians
- Returns dN/dE in m^-2 s^-1 sr^-1 GeV^-1

### Underground propagation: `Downbore_Analysis/muon_diff_energy_spectrum_underground.py`

The model treats the overburden as flat with a continuous energy loss, dE/dX = -a - bE:

| Quantity | Expression |
|---|---|
| Slant depth | X = h / cos(theta)  (h in m.w.e.) |
| Minimum surface energy to reach X | E_min = (a/b)(exp(bX) - 1) |
| Underground intensity | I(h, theta) = integral of the surface flux from E_min to infinity |

- The effective loss parameters are a = 0.25 GeV/m.w.e. and b = 0.36e-3 /m.w.e. With these values, the total flux agrees with published MUTE + daemonflux values to within about 4% at 2.0 to 2.9 km.w.e.
- The model is reliable from roughly 0.5 to 6 km.w.e. It does not account for local rock density or terrain.
- Depths in metres of rock are converted to m.w.e. using standard rock, 2.65 g/cm^3.

### Detector rate

For the upright hollow cylinder (scintillator tube), the projected area is

    A(theta) = pi (R_out^2 - R_in^2) cos(theta) + 2 R_out L sin(theta)

and the rate is

    N = 2 pi * integral I(h, theta) A(theta) sin(theta) dtheta

## Running

Each script can be run from any directory:

```bash
# Underground spectra, angular distributions, depth-intensity; writes PNGs to Downbore_Analysis/Plots/
python Downbore_Analysis/muon_diff_energy_spectrum_underground.py

# Muons per second / per day on the downbore tube at 250-2000 m; writes cylinder_muon_rate_vs_depth.txt
python Downbore_Analysis/cylinder_muon_rate_vs_depth.py

# Event rate, then median light yield (photoelectrons) per depth by Monte Carlo (~1-2 min)
python Downbore_Analysis/light_yield_monte_carlo.py
```

To change the detector geometry or depths:

- **`cylinder_muon_rate_vs_depth.py`:** edit `R_OUT`, `R_IN`, `LENGTH` (mm) and `RHO_ROCK`. Pass `depths` to `main()` to set the depths.
- **`light_yield_monte_carlo.py`:** edit `R, r, L`, `DEPTHS_M`, and the light-chain constants at the top of the file.

`light_yield_monte_carlo.py` is self-contained: it keeps its own copy of the surface flux and intensity code, so it can be shared as a single file. If you change the physics in `guan_model.py` or `muon_diff_energy_spectrum_underground.py`, update that copy too.

## Example output (current tube: R_out 37.5 mm, R_in 20.05 mm, L 2500 mm)

| Depth (m) | Depth (m.w.e.) | Muons/day |
|---:|---:|---:|
| 250 | 662 | 600.7 |
| 500 | 1325 | 69.7 |
| 750 | 1988 | 15.4 |
| 1000 | 2650 | 4.5 |
| 1250 | 3312 | 1.5 |
| 1500 | 3975 | 0.6 |
| 1750 | 4638 | 0.2 |
| 2000 | 5300 | 0.1 |
