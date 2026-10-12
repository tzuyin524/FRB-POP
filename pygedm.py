import pandas as pd
import numpy as np
from mwprop.nemod.NE2025 import ne2025
from astropy.coordinates import SkyCoord
from astropy import units as u
# --- Configuration ---
CSV_FILE = '/home/thsu/FRB_population/chimefrbcat2.csv'   # read from and written back to this same file

RA_COL = 'ra'      # RA column name
DEC_COL = 'dec'    # Dec column name
DM_COL = 'dm_fitb'      # observed DM column name

DEGREES = False    # True if RA/Dec are decimal degrees, False for hh:mm:ss / dd:mm:ss
DMD = 50           # dmd passed to ne2025

def dm_mw_ne2025(ra, dec, degrees=False, dmd=50):
    '''Return NE2025 DM_MW for one position (np.nan if it can't be computed).'''
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
        print(f'Could not compute DM for RA={ra}, Dec={dec}: {e}')
        return np.nan

df = pd.read_csv(CSV_FILE)

for col in (RA_COL, DEC_COL):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")

df['DM_ne2025'] = [
    round(dm_mw_ne2025(ra, dec, degrees=DEGREES, dmd=DMD), 2)
    for ra, dec in zip(df[RA_COL], df[DEC_COL])
]

if DM_COL in df.columns:
    df['DM_EG'] = (pd.to_numeric(df[DM_COL], errors='coerce') - df['DM_ne2025']).round(2)
else:
    print(f"Warning: DM column '{DM_COL}' not found, so DM_EG was not added. Available columns: {list(df.columns)}")

df.head()
df.to_csv(CSV_FILE, index=False)
