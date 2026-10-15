"""Command-line runner.

  python run_all.py                       # all telescopes, default parameters
  python run_all.py pks fast              # just these
  python run_all.py --set SNR_threshold=9 --set ALPHA_SCH=-1.5 --tag _test
  python run_all.py --set LUMINOSITY_FUNCTION=std --set USE_BEAM=False
"""
import argparse
from params import SimParams
from telescopes import ALL_KEYS, OUT_DIR, get_telescope
from core import run_survey


def _parse(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=ALL_KEYS)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _snr9")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    p = SimParams().with_overrides(**{k: _parse(v) for k, v in (s.split("=", 1) for s in a.set)})
    print(p)
    for key in a.keys:
        res = run_survey(get_telescope(key), p)
        print("  saved ->", res.save(a.out, a.tag))
