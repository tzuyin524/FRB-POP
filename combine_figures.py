from PIL import Image

# Paths to the two figures
left_path = "/home/thsu/FRB_population/DM_distribution_3tel_euc.png"
right_path = "/home/thsu/FRB_population/DM_distribution_3tel_beam.png"
out_path = "/home/thsu/FRB_population/DM_distribution_3tel_combined.png"

left = Image.open(left_path)
right = Image.open(right_path)

# Match heights (resize the shorter one to match, preserving aspect ratio)
target_height = max(left.height, right.height)

def resize_to_height(img, height):
    if img.height == height:
        return img
    ratio = height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, height), Image.LANCZOS)

left = resize_to_height(left, target_height)
right = resize_to_height(right, target_height)

combined_width = left.width + right.width
combined = Image.new("RGB", (combined_width, target_height), color="white")
combined.paste(left, (0, 0))
combined.paste(right, (left.width, 0))

combined.save(out_path, dpi=(300, 300))
print(f"Saved combined figure to {out_path}")