"""Frequency-dependent sky temperature loader (one implementation for every telescope)."""
import re
import numpy as np
import pandas as pd


def _norm(c):
    return str(c).lower().replace("_", "").replace(" ", "").strip()


def _to_Hz(f):
    f = np.asarray(f, dtype=float)
    return f * 1e6 if f.max() < 1e5 else f          # looks like MHz -> Hz


def find_freq_col(df, exclude=()):
    for col in df.columns:
        if col in exclude:
            continue
        s = str(col).lower().strip()
        if s.startswith(("freq", "nu")) or "hz" in s:
            return col
    return None


class TskyCurve:
    """Tsky(nu) in K.

    Accepted csv layouts
      long : a 'Tsky' column (+ optional frequency column freq/freq_MHz/nu/...), one row per frequency
      wide : columns named Tsky_<MHz> (e.g. Tsky_1050); several rows are averaged
    If there is no frequency information: rows are used directly when the count matches the
    requested frequencies, otherwise spread uniformly over the band.
    If the file has no Tsky at all, `fallback_K` (constant) is used when given.
    """

    def __init__(self, freqs_Hz, values, source=""):
        self.freqs_Hz = None if freqs_Hz is None else np.asarray(freqs_Hz, float)
        self.values = np.asarray(values, float)
        self.source = source
        if not np.all(np.isfinite(self.values)):
            raise ValueError(f"Tsky values contain NaN/inf ({source})")

    @classmethod
    def constant(cls, value_K, source="constant"):
        return cls(None, [float(value_K)], source)

    @classmethod
    def from_csv(cls, path, fallback_K=None):
        df = pd.read_csv(path)
        wide = [(col, float(m.group(1))) for col in df.columns
                for m in [re.match(r"(?i)^\s*tsky[_\s]*(\d+(?:\.\d+)?)", str(col))] if m]
        if wide:
            vals = df[[c for c, _ in wide]].values.astype(float)
            if vals.shape[0] > 1:
                print(f"  NOTE: {path} has {vals.shape[0]} rows; Tsky is averaged over them")
            return cls(_to_Hz([f for _, f in wide]), vals.mean(axis=0), path)
        cols = {_norm(c): c for c in df.columns}
        if "tsky" in cols:
            tcol = cols["tsky"]
        elif len(df.columns) == 1:
            tcol = df.columns[0]
        elif fallback_K is not None:
            print(f"  NOTE: no Tsky column in {path}; using constant {fallback_K} K")
            return cls.constant(fallback_K, path)
        else:
            raise ValueError(f"Cannot find a 'Tsky' column in {path}; columns = {list(df.columns)}")
        fcol = find_freq_col(df, exclude=(tcol,))
        freqs = _to_Hz(df[fcol].values) if fcol is not None else None
        return cls(freqs, df[tcol].values, path)

    @property
    def is_constant(self):
        return len(self.values) == 1 and self.freqs_Hz is None

    def at(self, nu_target):
        nu = np.atleast_1d(nu_target).astype(float)
        if self.is_constant:
            return np.full(nu.shape, self.values[0])
        if self.freqs_Hz is not None:
            o = np.argsort(self.freqs_Hz)
            return np.interp(nu, self.freqs_Hz[o], self.values[o])
        if len(nu) == len(self.values):
            return self.values.copy()
        pos = (nu - nu.min()) / max(nu.max() - nu.min(), 1e-30)
        return np.interp(pos, np.linspace(0, 1, len(self.values)), self.values)

    def check_against(self, nu_array):
        if self.freqs_Hz is None or self.is_constant:
            return
        if len(self.freqs_Hz) == len(nu_array):
            d = np.abs(np.sort(self.freqs_Hz) - np.sort(nu_array)).max() / 1e6
            print(f"  Tsky csv vs nu_array: {len(nu_array)} points, max |dnu| = {d:.3f} MHz")
        else:
            print(f"  WARNING: Tsky csv has {len(self.freqs_Hz)} rows but nu_array has "
                  f"{len(nu_array)}; interpolating")
