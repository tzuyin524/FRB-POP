# ============================================================================
# SCRIPT TO COMBINE ALL REDSHIFT CHUNKS
# ============================================================================

import numpy as np
from pathlib import Path
import glob

print("\nCombining all DM arrays from redshift chunks...")

# Find all DM files
dm_files = sorted(glob.glob('dm_detected_askap_z*.npy'))

if not dm_files:
    print("ERROR: No DM files found!")
    exit(1)

print(f"Found {len(dm_files)} files")

# Load and combine
all_dm_combined = []
for filename in dm_files:
    dm_array = np.load(filename)
    all_dm_combined.extend(dm_array)
    print(f"  {filename}: {len(dm_array)} points")

all_dm_combined = np.array(all_dm_combined)
print (all_dm_combined)

# Save combined array
output_file = 'dm_detected_askap_combined_full.npy'
np.save(output_file, all_dm_combined)

print(f"\nSuccessfully combined {len(dm_files)} chunks")
print(f"Total DM points: {len(all_dm_combined)}")
print(f"Saved to: {output_file}")

# Print statistics
print(f"\nDM Statistics:")
print(f"  Min: {all_dm_combined.min():.1f}")
print(f"  Max: {all_dm_combined.max():.1f}")
print(f"  Mean: {all_dm_combined.mean():.1f}")
print(f"  Median: {np.median(all_dm_combined):.1f}")
