"""Per-telescope configuration. This is the ONLY file that should differ between surveys.

Paths: set the environment variable FRB_DATA_DIR (default below) if the csv files live elsewhere.
"""
import os
from dataclasses import dataclass, field

from beams import (GaussianDishBeam, MultiBeamListDish, FastMultiBeamDish,
                   TiedRectBeam, TiedTriangleBeam)
from tsky import TskyCurve

DATA_DIR = os.environ.get("FRB_DATA_DIR", "/home/thsu/FRB_population")
OUT_DIR = os.environ.get("FRB_OUT_DIR", os.path.join(DATA_DIR, "DM_distribution"))


@dataclass
class Telescope:
    key: str                 # short name used by run_survey("...")
    label: str               # for plots
    out_name: str            # file stem: DM_distribution_<out_name>.npy
    D: float                 # dish diameter [m]
    eta: float               # aperture efficiency
    Trec: float              # receiver temperature [K]
    BW: float                # bandwidth [Hz]
    nu_cen: float            # centre frequency [Hz]
    n_comp: int              # number of frequency components
    n_chan: int              # number of channels (DM smearing)
    tsamp: float             # sampling time, LEGACY units (see note in README: used as tsamp*1e-3 s)
    beam: object             # beam model (beams.py)
    tsky_csv: str = None     # frequency-dependent Tsky; None -> constant Tsky_const
    Tsky_const: float = 3.0  # fallback sky temperature [K]
    color: str = "#1f77b4"
    overrides: dict = field(default_factory=dict)   # per-telescope SimParams overrides

    def tsky_curve(self):
        if self.tsky_csv is None:
            return TskyCurve.constant(self.Tsky_const)
        return TskyCurve.from_csv(self.tsky_csv, fallback_K=self.Tsky_const)


def _p(*parts):
    return os.path.join(DATA_DIR, *parts)


def build_telescopes():
    """Built lazily so a missing csv only breaks the telescope that needs it."""
    T = {}

    T["askap_fe"] = lambda: Telescope(
        "askap_fe", "ASKAP fly's eye", "askap_fe", D=12, eta=0.8, Trec=70, BW=336e6, nu_cen=1.32e9,
        n_comp=16, n_chan=2048, tsamp=1.265e-3, beam=GaussianDishBeam("single"),
        tsky_csv=_p("ASKAP_Tsky.csv"), color="#1f77b4")

    T["askap_ics"] = lambda: Telescope(
        "askap_ics", "ASKAP ICS", "askap_ics", D=12, eta=0.8, Trec=70, BW=336e6, nu_cen=1.32e9,
        n_comp=16, n_chan=2048, tsamp=1.265e-3, beam=GaussianDishBeam("sqrt_n", n_dish=36),
        tsky_csv=_p("ASKAP_Tsky.csv"), color="#17becf")

    T["pks"] = lambda: Telescope(
        "pks", "Parkes", "parkes", D=64, eta=0.7, Trec=28, BW=340e6, nu_cen=1.382e9,
        n_comp=16, n_chan=int(1024 * 340 / 400), tsamp=0.064e-3,
        beam=MultiBeamListDish([0.735] + [0.690] * 6 + [0.581] * 6),
        tsky_csv=_p("Parkes_Tsky.csv"), color="#ff7f0e")

    T["fast"] = lambda: Telescope(
        "fast", "FAST", "fast", D=300, eta=0.6, Trec=17, BW=400e6, nu_cen=1.25e9,
        n_comp=8, n_chan=2048, tsamp=196.608e-6, beam=FastMultiBeamDish(_p("FAST_beam_gain.csv")),
        tsky_csv=_p("FAST_Tsky.csv"), color="#2ca02c")

    # TODO: fast_plane.ipynb you uploaded is byte-identical to fast.ipynb (it even saves to
    # DM_distribution_fast.npy). Put whatever really differs for the plane survey here
    # (Tsky csv, sky area, Trec, tsamp, ...). Until then it is FAST with a different output name.
    T["fast_plane"] = lambda: Telescope(
        "fast_plane", "FAST plane", "fast_plane", D=300, eta=0.6, Trec=17, BW=400e6, nu_cen=1.25e9,
        n_comp=8, n_chan=2048, tsamp=196.608e-6, beam=FastMultiBeamDish(_p("FAST_beam_gain.csv")),
        tsky_csv=_p("FAST_Tsky_plane.csv"), color="#98df8a")

    T["meerkat_ib"] = lambda: Telescope(
        "meerkat_ib", "MeerKAT IB", "meerkat_ib", D=13.5, eta=0.70, Trec=20, BW=856e6, nu_cen=1.284e9,
        n_comp=16, n_chan=2048, tsamp=306.24e-6, beam=GaussianDishBeam("sqrt_n", n_dish=64),
        tsky_csv=_p("MeerKat_beams", "beam_parameters.csv"), color="#d62728")

    T["meerkat_cb"] = lambda: Telescope(
        "meerkat_cb", "MeerKAT CB", "meerkat_cb", D=13.5, eta=0.70, Trec=20, BW=856e6, nu_cen=1.284e9,
        n_comp=16, n_chan=2048, tsamp=306.24e-6,
        beam=TiedTriangleBeam(_p("MeerKat_beams", "beam_parameters.csv"), n_dishes=40),
        tsky_csv=_p("MeerKat_beams", "beam_parameters.csv"), color="#ff9896")

    T["dsa110"] = lambda: Telescope(
        "dsa110", "DSA-110", "dsa110", D=4.65, eta=0.65, Trec=22, BW=250e6, nu_cen=1.405e9,
        n_comp=9, n_chan=8192, tsamp=262.144e-3,
        beam=TiedRectBeam(_p("DSA110_beam_parameters.csv"), n_dishes=48, primary="dish",
                          no_freq_grid="csv_rows"),
        tsky_csv=_p("DSA110_beam_parameters.csv"), color="#8c564b")

    T["apertif"] = lambda: Telescope(
        "apertif", "Apertif", "apertif", D=25, eta=0.75, Trec=70, BW=300e6, nu_cen=1.37e9,
        n_comp=11, n_chan=1536, tsamp=0.08192e-3,
        beam=TiedRectBeam(_p("Apertif_beam_parameters.csv"), n_dishes=12, primary="apertif_empirical",
                          no_freq_grid="n_comp+1"),
        tsky_csv=_p("Apertif_beam_parameters.csv"), color="#9467bd")
    return T


_FACTORIES = build_telescopes()
ALL_KEYS = list(_FACTORIES)


def get_telescope(key):
    if key not in _FACTORIES:
        raise KeyError(f"Unknown telescope {key!r}; choose from {ALL_KEYS}")
    return _FACTORIES[key]()
