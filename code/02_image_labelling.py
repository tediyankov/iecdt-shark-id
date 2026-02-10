## Code for labelling cropped shark images
## Input: cropped shark images from 01_crop_sharks.py
## Output: CSV file with image paths and species labels

import cv2
import pandas as pd
import os
from pathlib import Path

## set paths
IMAGE_DIR = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/cropped_sharks"
OUTPUT_CSV = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/labels.csv"

# Species mapping
SPECIES = {
    '1': 'grey_reef_shark',
    '2': 'blacktip_reef_shark',
    '3': 'whitetip_reef_shark',
    '4': 'tawny_nurse_shark',
    '0': 'unclear/other',
    'q': 'quit'
}

def label_images():
    # getting all the images
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    
    # load existing labels if available
    labels = {}
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        labels = dict(zip(df['image_path'], df['species']))
        print(f"Loaded {len(labels)} existing labels")
    
    print("\n=== Shark Species Labeling ===")
    print("Press:")
    for key, species in SPECIES.items():
        print(f"  {key}: {species}")
    print("  s: skip")
    print("  b: go back")
    print("  ESC: save and quit")
    print("=" * 40)
    
    idx = 0
    while idx < len(images):
        img_name = images[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # skip if already labeled
        if img_path in labels:
            print(f"[{idx+1}/{len(images)}] Already labeled: {img_name} -> {labels[img_path]}")
            idx += 1
            continue
        
        # load and display image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_name}")
            idx += 1
            continue
        
        # resize if too large
        h, w = img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        cv2.imshow(f'Label Image [{idx+1}/{len(images)}]: {img_name}', img)
        
        # wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                save_labels(labels)
                return
            
            elif chr(key) in SPECIES and chr(key) != 'q':
                labels[img_path] = SPECIES[chr(key)]
                print(f"Labeled: {img_name} -> {SPECIES[chr(key)]}")
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
                save_labels(labels)
                return
        
        cv2.destroyAllWindows()
    
    save_labels(labels)
    print("\n✅ All images labeled!")

def save_labels(labels):
    df = pd.DataFrame([
        {'image_path': path, 'species': species}
        for path, species in labels.items()
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved {len(labels)} labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    label_images()