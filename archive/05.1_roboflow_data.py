import os
import pandas as pd
from pathlib import Path

# Configuration
ROBOFLOW_DIR = "/gws/nopw/j04/iecdt/shark_bruvs/roboflow2"
OUTPUT_CSV = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/roboflow_train_labels.csv"

# Species folders
SPECIES_FOLDERS = ['whitetip_reef_shark', 'blacktip_reef_shark', 'tawny_nurse_shark']

print("=" * 80)
print("Creating Roboflow Training Labels CSV")
print("=" * 80)
print()

# Collect all image paths and species labels
data = []

for species in SPECIES_FOLDERS:
    species_dir = os.path.join(ROBOFLOW_DIR, species)
    
    if not os.path.exists(species_dir):
        print(f"Warning: {species_dir} not found, skipping...")
        continue
    
    # Find all JPG files (case-insensitive)
    jpg_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        jpg_files.extend(Path(species_dir).glob(ext))
    
    print(f"{species:25s}: {len(jpg_files)} images")
    
    for img_path in jpg_files:
        data.append({
            'image_path': str(img_path),
            'species': species
        })

# Create DataFrame
df = pd.DataFrame(data)

print()
print(f"Total images collected: {len(df)}")
print()
print("Species distribution:")
print(df['species'].value_counts())
print()

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved to: {OUTPUT_CSV}")
print()
print("=" * 80)