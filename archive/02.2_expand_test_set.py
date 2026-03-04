
## code for expanding the test dataset with Sam and Greg's images from YouTube videos
## goal: test set with 600 images per class (with labels) + 2430 unclear/other (with labels) = 5430 total images in test set

import pandas as pd
import shutil
from pathlib import Path
import random

random.seed(42)

# config
EXISTING_LABELS = "labels.csv"
EXISTING_CROPPED = "cropped_sharks"
EXISTING_CROPPED_FULL_PATH = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/cropped_sharks"
GIGA_CROPPED = "/gws/nopw/j04/iecdt/shark_bruvs/giga_cropped"  
OUTPUT_DIR = "test_set"
OUTPUT_LABELS = "test_set_labels.csv"

# target counts (500 is what I set as a target but u can actually set it higher or lower. Aim for 100-200 per class)
TARGET_COUNTS = {
    "whitetip_reef_shark": 500,
    "grey_reef_shark": 500,
    "blacktip_reef_shark": 500,
    "tawny_nurse_shark": 500,
    "unclear/other": 500
}

# mapping from folder names to label names
FOLDER_TO_LABEL = {
    "Whitetip_Shark": "whitetip_reef_shark",
    "Greyreef_Shark": "grey_reef_shark",
    "Blacktip_Shark": "blacktip_reef_shark",
    "Nurse_Shark": "tawny_nurse_shark"
}

def main():
    # loading existing labels
    print("Loading existing labels...")
    df_existing = pd.read_csv(EXISTING_LABELS)
    print(f"Existing labels loaded: {len(df_existing)} images")
    print(df_existing['species'].value_counts())
    print()
    
    # creating output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # initialising list to collect all test set entries
    test_set_data = []
    
    # processing each species
    for species, target_count in TARGET_COUNTS.items():
        print(f"\n--- Processing {species} (target: {target_count}) ---")
        
        # getting existing images for this species
        existing_for_species = df_existing[df_existing['species'] == species].copy()
        existing_count = len(existing_for_species)
        print(f"Existing images: {existing_count}")
        
        if species == "unclear/other":
            # copuying all unclear/other images
            print(f"Copying all {existing_count} unclear/other images...")
            for _, row in existing_for_species.iterrows():
                src = Path(row['image_path'])
                # building full path if it's a relative path
                if not src.is_absolute():
                    src = Path(EXISTING_CROPPED_FULL_PATH) / src.name
                
                if src.exists():
                    dst = output_path / src.name
                    shutil.copy2(src, dst)
                    test_set_data.append({
                        'image_path': str(src),  # Use full source path
                        'species': species
                    })
            print(f"Copied {len([d for d in test_set_data if d['species'] == species])} images")
            continue
        
        # for other species, we need to combine existing + giga_cropped
        needed = target_count - existing_count
        print(f"Need {needed} more images from giga_cropped")
        
        # copying existing images
        for _, row in existing_for_species.iterrows():
            src = Path(row['image_path'])
            # building full path if it's a relative path
            if not src.is_absolute():
                src = Path(EXISTING_CROPPED_FULL_PATH) / src.name
            
            if src.exists():
                dst = output_path / src.name
                shutil.copy2(src, dst)
                test_set_data.append({
                    'image_path': str(src),  # Use full source path
                    'species': species
                })
        
        # finding corresponding giga_cropped folder
        giga_folder = None
        for folder_name, label_name in FOLDER_TO_LABEL.items():
            if label_name == species:
                giga_folder = Path(GIGA_CROPPED) / folder_name
                break
        
        if giga_folder is None or not giga_folder.exists():
            print(f"WARNING: Could not find giga_cropped folder for {species}")
            continue
        
        # getting all JPGs from giga_cropped folder
        giga_images = list(giga_folder.glob("*.jpg")) + list(giga_folder.glob("*.JPG"))
        print(f"Found {len(giga_images)} images in {giga_folder.name}")
        
        if len(giga_images) < needed:
            print(f"WARNING: Only {len(giga_images)} available, need {needed}")
            images_to_copy = giga_images
        else:
            # randomly sample the needed amount
            images_to_copy = random.sample(giga_images, needed)
        
        # copy selected images
        for img_path in images_to_copy:
            dst = output_path / img_path.name
            # handle potential name conflicts
            counter = 1
            while dst.exists():
                dst = output_path / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
            
            shutil.copy2(img_path, dst)
            test_set_data.append({
                'image_path': str(img_path), 
                'species': species
            })
        
        final_count = len([d for d in test_set_data if d['species'] == species])
        print(f"Final count for {species}: {final_count}")
    
    # create final labels CSV
    df_test = pd.DataFrame(test_set_data)
    df_test.to_csv(OUTPUT_LABELS, index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images in test set: {len(df_test)}")
    print("\nBreakdown by species:")
    print(df_test['species'].value_counts())
    print(f"\nTest set created in: {OUTPUT_DIR}/")
    print(f"Labels saved to: {OUTPUT_LABELS}")

if __name__ == "__main__":
    main()
