## Code for labelling shark behavior
## Input: labeled shark images (excluding unclear/other)
## Output: CSV file with image paths and behavior labels

import cv2
import pandas as pd
import os
from pathlib import Path

## set paths
SPECIES_CSV = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/test_set_labels.csv"
OUTPUT_CSV = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/behavior_labels.csv"
IMAGE_DIR = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/cropped_sharks"

# Behavior mapping
BEHAVIORS = {
    '1': 'cruising',
    '2': 'baiting',
    '0': 'unclear',
    'q': 'quit'
}

def label_behaviors():
    # Load species labels
    species_df = pd.read_csv(SPECIES_CSV)
    
    # Filter out unclear/other species
    species_df = species_df[species_df['species'] != 'unclear/other'].copy()
    print(f"Found {len(species_df)} images with clear species labels")
    
    # Load existing behavior labels if available
    behavior_labels = {}
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        behavior_labels = dict(zip(df['image_path'], df['behavior']))
        print(f"Loaded {len(behavior_labels)} existing behavior labels")
    
    print("\n=== Shark Behavior Labeling ===")
    print("Press:")
    for key, behavior in BEHAVIORS.items():
        print(f"  {key}: {behavior}")
    print("  s: skip")
    print("  b: go back")
    print("  ESC: save and quit")
    print("=" * 40)
    
    images = species_df['image_path'].tolist()
    idx = 0
    
    while idx < len(images):
        img_path = images[idx]
        img_name = os.path.basename(img_path)
        
        # Get species for context
        species = species_df[species_df['image_path'] == img_path]['species'].values[0]
        
        # Skip if already labeled
        if img_path in behavior_labels:
            print(f"[{idx+1}/{len(images)}] Already labeled: {img_name} ({species}) -> {behavior_labels[img_path]}")
            idx += 1
            continue
        
        # Load and display image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_name}")
            idx += 1
            continue
        
        # Resize if too large
        h, w = img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # Add species info to window title
        window_title = f'Behavior [{idx+1}/{len(images)}]: {img_name} - {species}'
        cv2.imshow(window_title, img)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                save_labels(behavior_labels)
                return
            
            elif chr(key) in BEHAVIORS and chr(key) != 'q':
                behavior_labels[img_path] = BEHAVIORS[chr(key)]
                print(f"Labeled: {img_name} ({species}) -> {BEHAVIORS[chr(key)]}")
                idx += 1
                break
            
            elif key == ord('s'):  # Skip
                idx += 1
                break
            
            elif key == ord('b'):  # Back
                if idx > 0:
                    idx -= 1
                break
            
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                save_labels(behavior_labels)
                return
        
        cv2.destroyAllWindows()
    
    save_labels(behavior_labels)
    print("\n✅ All images labeled!")

def save_labels(behavior_labels):
    df = pd.DataFrame([
        {'image_path': path, 'behavior': behavior}
        for path, behavior in behavior_labels.items()
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved {len(behavior_labels)} behavior labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    label_behaviors()