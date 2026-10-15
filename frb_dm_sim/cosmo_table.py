"""Telescope-independent redshift tables (computed once, shared by every survey)."""
import numpy as np

C_M_S = 3.0e8


class ZGrid:
    def __init__(self, p):
        self.p = p
        self.z = np.arange(p.z_min, p.z_max, p.dz)
        zmin, zmax = self.z - 0.5 * p.dz, self.z + 0.5 * p.dz
        if p.USE_COSMOLOGY:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=p.H0, Om0=p.Om0)
            self.dl_kpc = np.array([cosmo.luminosity_distance(z).value for z in self.z]) * 1000.0
            age_lo = np.array([cosmo.age(z).value for z in zmin])
            age_hi = np.array([cosmo.age(z).value for z in zmax])
            vol_hi = np.array([cosmo.comoving_volume(z).value for z in zmax])
            vol_lo = np.array([cosmo.comoving_volume(z).value for z in zmin])
            self.dt_yr = (age_lo - age_hi) * 1e9
            self.dvol_per_sr = (vol_hi - vol_lo) / (4.0 * np.pi)       # Mpc^3 / sr
        else:
            DH = 299792.458 / p.H0
            r, dr = DH * self.z, DH * p.dz
            self.dl_kpc = DH * self.z * 1000.0
            self.dvol_per_sr = r ** 2 * dr
            self.dt_yr = dr * 3.0857e22 / C_M_S / (3600.0 * 24.0 * 365.0)
            self.dt_yr = np.full_like(self.z, self.dt_yr)

    def dvol(self, i, area_deg2):
        return self.dvol_per_sr[i] * area_deg2 * (np.pi / 180.0) ** 2


_CACHE = {}


def get_zgrid(p):
    key = (p.dz, p.z_min, p.z_max, p.USE_COSMOLOGY, p.H0, p.Om0)
    if key not in _CACHE:
        _CACHE[key] = ZGrid(p)
    return _CACHE[key]
