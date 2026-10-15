"""Beam / gain models -- the only place where telescopes really differ.

Every model implements the same four steps used by core.run_survey:

    frequency_grid(tel)           -> (nu_array [Hz], n_comp)
    prepare(tel, nu_array, nu0)   -> dict(area_deg2, f_area, G0_max)   (also caches what draw/gain need)
    draw(n)                       -> geometry for n FRBs   (consumes RNG *before* the luminosity draw)
    gain(geom, n, use_beam)       -> G_nu (n, n_nu) [K/Jy], gain_ratio_nu0 (n,), ok (n,) bool

The RNG call order inside draw()/gain() reproduces the original notebooks exactly, so with the
same seed the new code returns the same numbers as the old ones.
"""
import numpy as np
import pandas as pd

K_B = 1.38e-23
C = 3.0e8


def dish_G0(D, eta):
    """Single-dish on-axis gain in K/Jy."""
    return np.pi * D * D / 4 * eta / 2.0 / K_B * 1e-26


def _default_grid(tel):
    nu_min, nu_max = tel.nu_cen - tel.BW / 2, tel.nu_cen + tel.BW / 2
    return np.linspace(nu_min, nu_max, tel.n_comp + 1), tel.n_comp


_FREQ_NAMES = ("freq", "frequency", "nu", "freq_hz", "freq_mhz", "nu_hz", "nu_mhz")


def _beam_freq_col(df):
    return next((c for c in df.columns if str(c).lower() in _FREQ_NAMES), None)


def _col_freqs_Hz(df, col):
    f = df[col].values.astype(float)
    return f * 1e6 if f.max() < 1e5 else f


def _interp_on_csv(nu_target, df, freq_col, raw):
    """Resample a per-row csv quantity onto nu_target (Hz)."""
    nu_target = np.atleast_1d(nu_target).astype(float)
    if freq_col is not None:
        f = _col_freqs_Hz(df, freq_col)
        o = np.argsort(f)
        return np.interp(nu_target, f[o], raw[o])
    if len(nu_target) == len(raw):
        return raw.copy()
    pos = (nu_target - nu_target.min()) / max(nu_target.max() - nu_target.min(), 1e-30)
    return np.interp(pos, np.linspace(0, 1, len(raw)), raw)


# =============================================================================
# 1. Single dish with a Gaussian primary beam
#    ASKAP fly's eye / ICS, Parkes, FAST, MeerKAT incoherent
# =============================================================================
class GaussianDishBeam:
    """G(theta, nu) = G0 * exp(-C theta^2 nu^2), FRBs uniform in a disc of radius 2*FWHM(nu_min)
    and only counted inside 4 theta_half-power.

    gain_scale: 'single' -> G0 = one dish (ASKAP fly's eye)
                'sqrt_n' -> G0 = sqrt(n_dish) * single (incoherent sum: ASKAP ICS, MeerKAT IB)
    """

    def __init__(self, gain_scale="single", n_dish=1):
        self.gain_scale = gain_scale
        self.n_dish = n_dish

    # -- hooks overridden by the Parkes / FAST variants ----------------------
    def frequency_grid(self, tel):
        return _default_grid(tel)

    def _on_axis_gain(self, tel, nu_array, nu0):
        """returns (G0 used when drawing per-FRB, G0_max used for L_min(z))"""
        G0 = dish_G0(tel.D, tel.eta)
        if self.gain_scale == "sqrt_n":
            G0 = np.sqrt(self.n_dish) * G0
        elif self.gain_scale != "single":
            raise ValueError(self.gain_scale)
        return G0, G0

    def _G0_per_frb(self, n):
        return self.G0

    # -- common machinery ----------------------------------------------------
    def prepare(self, tel, nu_array, nu0):
        D = tel.D
        nu_min = tel.nu_cen - tel.BW / 2
        self.nu_array, self.nu0 = nu_array, nu0
        self.coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * C ** 2)
        self.theta_max = 2 * 1.22 * C / D / nu_min
        fwhm_deg = 1.22 * (C / nu_min) / D * (180 / np.pi)
        area = np.pi * (2.0 * fwhm_deg) ** 2
        self.theta_hp = np.sqrt(np.log(2) / (self.coeff * nu0 ** 2))
        self.G0, G0_max = self._on_axis_gain(tel, nu_array, nu0)
        self.desc = f"G0={np.max(G0_max):.4g}, theta_max={self.theta_max:.6f} rad"
        return dict(area_deg2=area, f_area=1.0, G0_max=float(np.max(G0_max)))

    def draw(self, n):
        theta_b = self.theta_max * np.sqrt(np.random.random(n))
        return theta_b, theta_b <= 4.0 * self.theta_hp

    def gain(self, geom, n, use_beam):
        theta_b, angle_ok = geom
        theta_for_gain = np.where(angle_ok, theta_b, 0.0) if use_beam else np.zeros(n)
        G0_b = self._G0_per_frb(n)
        G_nu = G0_b * np.exp(-(self.coeff * theta_for_gain[:, None] ** 2) * self.nu_array[None, :] ** 2)
        ratio = np.exp(-self.coeff * theta_b ** 2 * self.nu0 ** 2) if use_beam else np.ones(n)
        return G_nu, ratio, angle_ok


class MultiBeamListDish(GaussianDishBeam):
    """Parkes 13-beam receiver: on-axis gain (K/Jy) drawn per FRB from a list, frequency flat."""

    def __init__(self, gains):
        super().__init__("single")
        self.gains = list(gains)

    def _on_axis_gain(self, tel, nu_array, nu0):
        return None, max(self.gains)

    def _G0_per_frb(self, n):
        return np.random.choice(self.gains, size=n)[:, None]


class FastMultiBeamDish(GaussianDishBeam):
    """FAST 19-beam receiver: per-beam gain measured at several frequencies (csv with Gain_<MHz> columns).
    The simulation runs on the csv frequencies and draws one beam per FRB."""

    def __init__(self, gain_csv):
        super().__init__("single")
        self.gain_csv = gain_csv
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        df = pd.read_csv(self.gain_csv)
        cols = [c for c in df.columns if str(c).startswith("Gain_")]
        self.freqs_measured = np.array([int(c.split("_")[1]) for c in cols]) * 1e6
        self.gain_matrix = df[cols].values.astype(float)                  # (n_beams, n_freq)
        self._loaded = True

    def _cache_for(self, nu):
        from scipy.interpolate import interp1d
        f = interp1d(self.freqs_measured, self.gain_matrix, kind="linear", axis=1,
                     bounds_error=False, fill_value="extrapolate")
        return np.clip(f(np.atleast_1d(nu)), 0, self.gain_matrix.max(axis=1, keepdims=True))

    def frequency_grid(self, tel):
        self._load()
        return self.freqs_measured.copy(), len(self.freqs_measured)       # one component per csv gain column

    def _on_axis_gain(self, tel, nu_array, nu0):
        self._load()
        self.beam_cache = self._cache_for(nu_array)                       # (n_beams, n_nu)
        return None, float(self._cache_for(np.array([nu0])).max())

    def _G0_per_frb(self, n):
        return self.beam_cache[np.random.randint(self.beam_cache.shape[0], size=n)]


# =============================================================================
# 2. Coherent arrays: primary beam x tied-array beam
# =============================================================================
class _PrimaryDish:
    """Gaussian dish primary beam with the same C*theta^2*nu^2 form as the single-dish notebooks."""

    def setup(self, tel, nu_array, nu0):
        D = tel.D
        nu_min = tel.nu_cen - tel.BW / 2
        self.coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * C ** 2)
        self.nu_array, self.nu0 = nu_array, nu0
        fwhm_rad = 1.22 * (C / nu_min) / D
        fwhm_deg = fwhm_rad * (180 / np.pi)
        area = np.pi * (2.0 * fwhm_deg) ** 2
        self.theta_fov = 2.0 * fwhm_rad
        return area

    def draw_radius(self, area):
        return self.theta_fov, 1.0                                        # (theta_draw, f_area)

    def envelope(self, theta, nu):
        return np.exp(-(self.coeff * theta[:, None] ** 2) * nu[None, :] ** 2)

    def envelope0(self, theta):
        return np.exp(-self.coeff * theta ** 2 * self.nu0 ** 2)


class _PrimaryEmpirical:
    """Apertif primary beam: FWHM(nu) = -3e-8 nu + 77 arcmin, Gaussian."""

    @staticmethod
    def sigma(nu):
        fwhm_arcmin = -3e-8 * np.asarray(nu, dtype=float) + 77.0
        return fwhm_arcmin * (np.pi / (180.0 * 60.0)) / 2.355

    def setup(self, tel, nu_array, nu0):
        D = tel.D
        nu_min = tel.nu_cen - tel.BW / 2
        self.nu_array, self.nu0 = nu_array, nu0
        fwhm_deg = 1.22 * (C / nu_min) / D * (180 / np.pi)
        area = np.pi * (2.0 * fwhm_deg) ** 2
        self.sig_arr, self.sig0 = self.sigma(nu_array), self.sigma(nu0)
        return area

    def draw_radius(self, area, THETA_CUT_FWHM=2.0):
        theta_fov = np.sqrt(area / np.pi) * np.pi / 180.0
        fwhm_min = 2.355 * self.sig_arr.max()
        theta_draw = min(theta_fov, THETA_CUT_FWHM * fwhm_min)
        return theta_draw, (theta_draw / theta_fov) ** 2

    def envelope(self, theta, nu):
        return np.exp(-0.5 * (theta[:, None] / self.sig_arr[None, :]) ** 2)

    def envelope0(self, theta):
        return np.exp(-0.5 * (theta / self.sig0) ** 2)


class TiedRectBeam:
    """Two tied beams separated along x (DSA-110, Apertif): csv with sigma_avg, fwhm_avg, beam_sep
    (arcsec) [+ freq column, Tsky column]. FRBs are drawn uniformly inside one beam cell
    (x in +-sep/2, y in +-4 sigma_centre) and take the max over the two beams.

    primary = 'dish' (DSA-110) or 'apertif_empirical'.
    n_comp / grid: if the csv has a frequency column the simulation runs on exactly those
    frequencies (n_comp := number of rows). Without one, DSA-110 uses one point per csv row,
    Apertif uses n_comp+1 uniformly spaced points (`no_freq_grid`).
    """

    def __init__(self, csv, n_dishes, primary="dish", no_freq_grid="csv_rows"):
        self.csv, self.n_dishes = csv, n_dishes
        self.primary = _PrimaryDish() if primary == "dish" else _PrimaryEmpirical()
        self.no_freq_grid = no_freq_grid
        df = pd.read_csv(csv)
        self.df = df
        self.sigma_raw = df["sigma_avg"].values.astype(float)             # arcsec
        self.sep = float(df["beam_sep"].iloc[0])                          # arcsec
        sig_c = self.sigma_raw[len(self.sigma_raw) // 2]
        self.y_half = 4.0 * sig_c
        self.positions = np.array([[-self.sep / 2.0, 0.0], [self.sep / 2.0, 0.0]])
        self.fcol = _beam_freq_col(df)

    def frequency_grid(self, tel):
        if self.fcol is not None:
            nu = np.sort(_col_freqs_Hz(self.df, self.fcol))
            if len(nu) != tel.n_comp:
                print(f"  NOTE: n_comp {tel.n_comp} -> {len(nu)} to match the csv rows")
            return nu, len(nu)
        if self.no_freq_grid == "csv_rows":
            nu_min, nu_max = tel.nu_cen - tel.BW / 2, tel.nu_cen + tel.BW / 2
            nu = np.linspace(nu_min, nu_max, len(self.sigma_raw))
            return nu, len(nu)
        return _default_grid(tel)

    def _tied_response(self, x, y, sigma):
        sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        d2 = ((x[:, None] - self.positions[None, :, 0]) ** 2
              + (y[:, None] - self.positions[None, :, 1]) ** 2)
        resp = np.exp(-0.5 * d2[:, :, None] / sigma[None, None, :] ** 2)
        return resp.max(axis=1)

    def prepare(self, tel, nu_array, nu0):
        area = self.primary.setup(tel, nu_array, nu0)
        self.theta_draw, f_area = self.primary.draw_radius(area)
        self.G0 = self.n_dishes * dish_G0(tel.D, tel.eta)
        self.sig_nu = _interp_on_csv(nu_array, self.df, self.fcol, self.sigma_raw)
        self.sig_0 = _interp_on_csv(np.array([nu0]), self.df, self.fcol, self.sigma_raw)
        self.nu_array = nu_array
        self.desc = f"G0={self.G0:.4e} (n_dishes={self.n_dishes}), draw radius {np.degrees(self.theta_draw):.3f} deg"
        return dict(area_deg2=area, f_area=f_area, G0_max=self.G0)

    def draw(self, n):
        theta = self.theta_draw * np.sqrt(np.random.random(n))
        x = np.random.uniform(-self.sep / 2.0, self.sep / 2.0, n)
        y = np.random.uniform(-self.y_half, self.y_half, n)
        return theta, x, y

    def gain(self, geom, n, use_beam):
        theta, x, y = geom
        if use_beam:
            G = self.G0 * self.primary.envelope(theta, self.nu_array) * self._tied_response(x, y, self.sig_nu)
            ratio = self.primary.envelope0(theta) * self._tied_response(x, y, self.sig_0)[:, 0]
        else:
            G, ratio = self.G0 * np.ones((n, len(self.nu_array))), np.ones(n)
        return G, ratio, np.ones(n, dtype=bool)


class TiedTriangleBeam:
    """Three tied beams on a triangle (MeerKAT coherent): csv with beam_sep, sigma_x, sigma_y
    [+ freq column, Tsky column], in BEAM_UNIT_DEG degrees. FRBs are drawn only inside the tiling
    circle; the rate is scaled by (theta_tiling/theta_max)^2 (identical to rejecting the rest)."""

    def __init__(self, csv, n_dishes, theta_tiling_deg=0.3119373, beam_unit_deg=1.0):
        self.csv, self.n_dishes = csv, n_dishes
        self.theta_tiling = np.pi * theta_tiling_deg / 180
        self.unit = beam_unit_deg
        df = pd.read_csv(csv)
        self.df = df
        sep = df["beam_sep"].iloc[0]
        R = sep / np.sqrt(3)
        self.positions = np.array([[0, R], [-sep / 2, -R / 2], [sep / 2, -R / 2]])
        self.sx_raw = df["sigma_x"].values.astype(float)
        self.sy_raw = df["sigma_y"].values.astype(float)
        self.fcol = _beam_freq_col(df)
        self.primary = _PrimaryDish()

    def frequency_grid(self, tel):
        return _default_grid(tel)

    def _response(self, x, y, sx, sy):
        out = np.empty((len(x), len(self.positions), len(sx)))
        for i, (bx, by) in enumerate(self.positions):
            out[:, i, :] = np.exp(-0.5 * (((x - bx)[:, None] / sx[None, :]) ** 2
                                          + ((y - by)[:, None] / sy[None, :]) ** 2))
        return out.max(axis=1)

    def prepare(self, tel, nu_array, nu0):
        D = tel.D
        nu_min = tel.nu_cen - tel.BW / 2
        self.primary.setup(tel, nu_array, nu0)
        theta_max = 2 * 1.22 * C / D / nu_min
        fwhm_deg = 1.22 * (C / nu_min) / D * (180 / np.pi)
        area = np.pi * (2.0 * fwhm_deg) ** 2
        self.theta_draw = min(self.theta_tiling, theta_max)
        f_area = (self.theta_draw / theta_max) ** 2
        self.G0 = self.n_dishes * dish_G0(D, tel.eta)
        self.sx_nu = _interp_on_csv(nu_array, self.df, self.fcol, self.sx_raw)
        self.sy_nu = _interp_on_csv(nu_array, self.df, self.fcol, self.sy_raw)
        self.sx_0 = _interp_on_csv(np.array([nu0]), self.df, self.fcol, self.sx_raw)
        self.sy_0 = _interp_on_csv(np.array([nu0]), self.df, self.fcol, self.sy_raw)
        self.nu_array = nu_array
        self.desc = (f"G0={self.G0:.4e} (n_dishes={self.n_dishes}), tiled fraction of disc {f_area:.4f}")
        return dict(area_deg2=area, f_area=f_area, G0_max=self.G0)

    def draw(self, n):
        theta = self.theta_draw * np.sqrt(np.random.random(n))
        phi = 2 * np.pi * np.random.random(n)
        scale = (180.0 / np.pi) / self.unit
        return theta * np.cos(phi) * scale, theta * np.sin(phi) * scale, theta

    def gain(self, geom, n, use_beam):
        x, y, theta = geom
        if use_beam:
            G = self.G0 * self.primary.envelope(theta, self.nu_array) * self._response(x, y, self.sx_nu, self.sy_nu)
            ratio = self.primary.envelope0(theta) * self._response(x, y, self.sx_0, self.sy_0)[:, 0]
        else:
            G, ratio = self.G0 * np.ones((n, len(self.nu_array))), np.ones(n)
        return G, ratio, np.ones(n, dtype=bool)
