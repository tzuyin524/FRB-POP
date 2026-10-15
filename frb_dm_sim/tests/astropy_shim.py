"""Minimal FlatLambdaCDM stand-in (no radiation, like astropy's default) so the equivalence test can run
where astropy isn't installed. Used for BOTH the old notebook code and the new package."""
import sys, types
import numpy as np
from scipy.integrate import quad
import scipy.integrate as si

class _Q:  # mimics astropy Quantity with .value
    def __init__(self, v): self.value = v

class FlatLambdaCDM:
    def __init__(self, H0, Om0):
        self.H0 = _Q(float(H0)); self.Om0 = Om0; self.Ol0 = 1 - Om0
        self.DH = 299792.458 / H0
    def E(self, z): return np.sqrt(self.Om0 * (1 + z) ** 3 + self.Ol0)
    def _dc(self, z): return self.DH * quad(lambda x: 1 / self.E(x), 0, z, epsabs=1e-13, epsrel=1e-13)[0]
    def luminosity_distance(self, z): return _Q((1 + z) * self._dc(z))
    def comoving_volume(self, z): return _Q(4 / 3 * np.pi * self._dc(z) ** 3)
    def age(self, z):
        Gyr = 977.792 / self.H0.value      # 1/H0 in Gyr
        f = lambda x: 1 / ((1 + x) * self.E(x))
        return _Q(Gyr * quad(f, z, np.inf, epsabs=1e-13, epsrel=1e-13)[0])

def install():
    ap = types.ModuleType("astropy"); cos = types.ModuleType("astropy.cosmology"); un = types.ModuleType("astropy.units")
    cos.FlatLambdaCDM = FlatLambdaCDM
    cos.Planck18 = FlatLambdaCDM(67.66, 0.3097)
    cos.z_at_value = lambda *a, **k: None
    un.u = None
    ap.cosmology, ap.units = cos, un
    sys.modules.update({"astropy": ap, "astropy.cosmology": cos, "astropy.units": un})
    # old notebooks import removed scipy names
    si.simps = si.simpson; si.cumtrapz = si.cumulative_trapezoid
    try:
        import sympy  # noqa
    except ImportError:
        sys.modules["sympy"] = types.ModuleType("sympy")
