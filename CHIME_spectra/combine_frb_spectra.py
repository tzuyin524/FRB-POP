import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Paths
# ============================================================

baseband_dir = "/home/thsu/FRB_population/CHIME_spectra/baseband_spectra"
cat1_dir = "/home/thsu/FRB_population/CHIME_spectra/chimecat1_spectra"

csv_file = "/home/thsu/FRB_population/CHIME_spectra/combined_alpha_sep.csv"

output_dir = "/home/thsu/FRB_population/CHIME_spectra/combine_spectra"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(csv_file)

# ---- Change this if your FRB column has a different name ----
FRB_COLUMN = "frb_name"

if FRB_COLUMN not in df.columns:
    print("Available columns:", df.columns.tolist())
    raise ValueError(
        f"Column '{FRB_COLUMN}' not found. "
        "Please replace FRB_COLUMN with the correct column name."
    )

if "sep_deg" not in df.columns:
    raise ValueError("Column 'sep_deg' not found in CSV.")

# Dictionary: FRB -> sep_deg
sep_dict = dict(zip(df[FRB_COLUMN].astype(str), df["sep_deg"]))

# ============================================================
# Extract FRB name from filename
# ============================================================

def extract_frb(filename):
    """
    Extract FRB name such as FRB20181209A
    """
    match = re.search(r"(FRB\d{8}[A-Za-z]+)", filename)
    return match.group(1) if match else None


# ============================================================
# Collect image files
# ============================================================

baseband_files = {}
cat1_files = {}

for file in os.listdir(baseband_dir):
    if file.lower().endswith(".png"):
        frb = extract_frb(file)
        if frb:
            baseband_files[frb] = os.path.join(baseband_dir, file)

for file in os.listdir(cat1_dir):
    if file.lower().endswith(".png"):
        frb = extract_frb(file)
        if frb:
            cat1_files[frb] = os.path.join(cat1_dir, file)

# Union of all FRBs
all_frbs = sorted(set(baseband_files) | set(cat1_files))

print(f"Found {len(all_frbs)} FRBs.")

# ============================================================
# Create combined figures
# ============================================================

for i, frb in enumerate(all_frbs, start=1):

    left_img = baseband_files.get(frb)
    right_img = cat1_files.get(frb)

    # Get separation
    sep = sep_dict.get(frb)

    if pd.isna(sep):
        sep_text = "sep_deg = N/A"
    else:
        sep_text = f"sep_deg = {float(sep):.4f}°"

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"{frb}    |    {sep_text}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # --------------------------------------------------------
    # LEFT : Baseband
    # --------------------------------------------------------
    ax = axes[0]

    if left_img and os.path.exists(left_img):
        img = Image.open(left_img)
        ax.imshow(img)
        ax.set_title("Baseband", fontsize=14)
    else:
        ax.text(
            0.5,
            0.5,
            "Missing\nBaseband",
            ha="center",
            va="center",
            fontsize=16,
        )

    ax.axis("off")

    # --------------------------------------------------------
    # RIGHT : CHIME cat1
    # --------------------------------------------------------
    ax = axes[1]

    if right_img and os.path.exists(right_img):
        img = Image.open(right_img)
        ax.imshow(img)
        ax.set_title("CHIME cat1", fontsize=14)
    else:
        ax.text(
            0.5,
            0.5,
            "Missing\nCHIME cat1",
            ha="center",
            va="center",
            fontsize=16,
        )

    ax.axis("off")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    outfile = os.path.join(output_dir, f"{frb}_combined.png")
    plt.savefig(outfile, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[{i:3d}/{len(all_frbs)}] Saved {outfile}")

print("\nDone!")
print(f"Combined figures saved to:\n{output_dir}")