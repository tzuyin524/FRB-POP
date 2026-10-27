#!/usr/bin/env python
"""
Add NE2025 Milky Way DM (DM_ne2025) and extragalactic DM (DM_EG = DM - DM_ne2025)
columns to an FRB CSV catalogue.

Example:
    python dm_ne2025_pipeline.py
    python dm_ne2025_pipeline.py --csv frbs.csv --degrees
    python dm_ne2025_pipeline.py --csv frbs.csv --output frbs_with_dm.csv --dmd 50
"""
import argparse
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from mwprop.nemod.NE2025 import ne2025


def dm_mw_ne2025(ra, dec, degrees=False, dmd=50):
    """Return NE2025 DM_MW for one position (np.nan if it can't be computed)."""
    try:
        if degrees:
            c = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        else:
            ra_str = str(ra).replace(' ', '')
            dec_str = str(dec).replace(' ', '')
            c = SkyCoord(f'{ra_str} {dec_str}', unit=(u.hourangle, u.deg))

        _, modouts, _, _ = ne2025(
            ldeg=c.galactic.l.value,
            bdeg=c.galactic.b.value,
            dmd=dmd,
            ndir=-1,
            classic=False,
        )
        return float(modouts['DM'])
    except Exception as e:
        print(f'Could not compute DM for RA={ra}, Dec={dec}: {e}', file=sys.stderr)
        return np.nan


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Compute NE2025 Milky Way DM and extragalactic DM for an FRB CSV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--csv', default='frbs.csv', help='Input CSV file')
    p.add_argument('--output', default=None,
                   help='Output CSV file (default: overwrite the input CSV)')
    p.add_argument('--ra-col', default='RA', help='RA column name')
    p.add_argument('--dec-col', default='Dec', help='Dec column name')
    p.add_argument('--dm-col', default='DM', help='Observed DM column name')
    p.add_argument('--degrees', action='store_true',
                   help='RA/Dec are decimal degrees (default: hh:mm:ss / dd:mm:ss)')
    p.add_argument('--dmd', type=float, default=50, help='dmd passed to ne2025')
    p.add_argument('--mw-col', default='DM_ne2025', help='Output column for MW DM')
    
    p.add_argument('--eg-col', default='DM_EG', help='Output column for extragalactic DM')
    p.add_argument('--decimals', type=int, default=2, help='Rounding decimals')
    return p.parse_args(argv)


def run(args):
    df = pd.read_csv(args.csv)

    for col in (args.ra_col, args.dec_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")

    df[args.mw_col] = [
        round(dm_mw_ne2025(ra, dec, degrees=args.degrees, dmd=args.dmd), args.decimals)
        for ra, dec in zip(df[args.ra_col], df[args.dec_col])
    ]

    if args.dm_col in df.columns:
        df[args.eg_col] = (
            pd.to_numeric(df[args.dm_col], errors='coerce') - df[args.mw_col]
        ).round(args.decimals)
    else:
        print(f"Warning: DM column '{args.dm_col}' not found, so {args.eg_col} was not added. "
              f"Available columns: {list(df.columns)}", file=sys.stderr)

    out = args.output or args.csv
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows to {out}')
    print(df.head())
    return df


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()