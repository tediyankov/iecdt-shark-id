
## code for creating a fine tuning dataset and a test set from BRUVS labelled videos and Olga's Roboflow data

import pandas as pd
import os
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
# Input files
TEST_SET_CSV = "./labels.csv"
EXTERNAL_CSV = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/roboflow_train_labels.csv" 

# Base directories
BASE_DIR = "."
EXTERNAL_BASE_DIR = "/gws/nopw/j04/iecdt/shark_bruvs/roboflow2" 

# Output directory
OUTPUT_DIR = "./data"
FINETUNE_DIR = os.path.join(OUTPUT_DIR, "finetune")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

os.makedirs(FINETUNE_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Target classes and sampling strategy
TARGET_CLASSES = ['grey_reef_shark', 'blacktip_reef_shark', 'whitetip_reef_shark', 'tawny_nurse_shark']

FINETUNE_SAMPLES = {
    'grey_reef_shark': {'my_data': 200, 'external': 0},      # No external grey reef
    'blacktip_reef_shark': {'my_data': 100, 'external': 100},
    'whitetip_reef_shark': {'my_data': 100, 'external': 100},
    'tawny_nurse_shark': {'my_data': 100, 'external': 100}
}

# ============================================================================
# STEP 1: Load Datasets
# ============================================================================
print("=" * 80)
print("Creating Fine-tuning and Test Splits")
print("=" * 80)
print()

print("Loading datasets...")
my_data = pd.read_csv(TEST_SET_CSV)
external_data = pd.read_csv(EXTERNAL_CSV)

print(f"My data: {len(my_data)} images")
print(f"My data distribution:\n{my_data['species'].value_counts()}\n")

print(f"External data: {len(external_data)} images")
print(f"External data distribution:\n{external_data['species'].value_counts()}\n")

# ============================================================================
# STEP 2: Create Fine-tuning Dataset
# ============================================================================
print("=" * 80)
print("STEP 2: Creating Fine-tuning Dataset")
print("=" * 80)
print()

finetune_samples = []
used_indices_my_data = []

for cls in TARGET_CLASSES:
    n_my = FINETUNE_SAMPLES[cls]['my_data']
    n_ext = FINETUNE_SAMPLES[cls]['external']
    
    print(f"\n{cls}:")
    print(f"  Sampling {n_my} from my data, {n_ext} from external")
    
    # Sample from my data
    my_cls_data = my_data[my_data['species'] == cls]
    
    if len(my_cls_data) < n_my:
        print(f"  ⚠️  Warning: Only {len(my_cls_data)} samples available in my data, using all")
        n_my = len(my_cls_data)
    
    my_samples = my_cls_data.sample(n=n_my, random_state=42)
    used_indices_my_data.extend(my_samples.index.tolist())
    
    # Add source column
    my_samples = my_samples.copy()
    my_samples['source'] = 'my_data'
    finetune_samples.append(my_samples)
    
    print(f"  ✅ Sampled {len(my_samples)} from my data")
    
    # sample from external data if needed
    if n_ext > 0:
        ext_cls_data = external_data[external_data['species'] == cls]
        
        if len(ext_cls_data) < n_ext:
            print(f"  ⚠️  Warning: Only {len(ext_cls_data)} samples available in external, using all")
            n_ext = len(ext_cls_data)
        
        ext_samples = ext_cls_data.sample(n=n_ext, random_state=42)
        ext_samples = ext_samples.copy()
        ext_samples['source'] = 'external'
        finetune_samples.append(ext_samples)
        
        print(f"  ✅ Sampled {len(ext_samples)} from external")

# Combine all fine-tuning samples
finetune_df = pd.concat(finetune_samples, ignore_index=True)

print(f"\n{'=' * 80}")
print(f"Fine-tuning dataset created:")
print(f"  Total samples: {len(finetune_df)}")
print(f"  Distribution by class:\n{finetune_df['species'].value_counts()}")
print(f"  Distribution by source:\n{finetune_df['source'].value_counts()}")
print()

# ============================================================================
# STEP 3: Create Test Dataset (my data minus fine-tuning samples)
# ============================================================================
print("=" * 80)
print("STEP 3: Creating Test Dataset")
print("=" * 80)
print()

# Remove used samples from my data
test_df = my_data.drop(used_indices_my_data).copy()
test_df['source'] = 'my_data'

print(f"Test dataset created:")
print(f"  Total samples: {len(test_df)}")
print(f"  Distribution by class:\n{test_df['species'].value_counts()}")
print()

# ============================================================================
# STEP 4: Save CSV Files
# ============================================================================
print("=" * 80)
print("STEP 4: Saving Dataset Metadata")
print("=" * 80)
print()

# Save fine-tuning dataset
finetune_csv_path = os.path.join(OUTPUT_DIR, "finetune_labels.csv")
finetune_df.to_csv(finetune_csv_path, index=False)
print(f"✅ Saved fine-tuning labels: {finetune_csv_path}")

# Save test dataset
test_csv_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
test_df.to_csv(test_csv_path, index=False)
print(f"✅ Saved test labels: {test_csv_path}")

# Create detailed split summary
summary = []
for cls in TARGET_CLASSES:
    finetune_count = len(finetune_df[finetune_df['species'] == cls])
    test_count = len(test_df[test_df['species'] == cls])
    
    finetune_my = len(finetune_df[(finetune_df['species'] == cls) & (finetune_df['source'] == 'my_data')])
    finetune_ext = len(finetune_df[(finetune_df['species'] == cls) & (finetune_df['source'] == 'external')])
    
    summary.append({
        'class': cls,
        'finetune_total': finetune_count,
        'finetune_my_data': finetune_my,
        'finetune_external': finetune_ext,
        'test_total': test_count,
        'original_total': finetune_count + test_count
    })

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTPUT_DIR, "split_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✅ Saved split summary: {summary_path}")

print("\nSplit Summary:")
print(summary_df.to_string(index=False))
print()

# ============================================================================
# STEP 5: Optional - Copy Images to Organized Directories
# ============================================================================
print("=" * 80)
print("STEP 5: Copying Images (Optional)")
print("=" * 80)
print()

copy_images = input("Do you want to copy images to organized directories? (yes/no): ").lower().strip()

if copy_images in ['yes', 'y']:
    print("\nCopying fine-tuning images...")
    
    for idx, row in finetune_df.iterrows():
        # Determine source directory
        if row['source'] == 'my_data':
            src_path = os.path.join(BASE_DIR, row['image_path'])
        else:
            src_path = os.path.join(EXTERNAL_BASE_DIR, row['image_path'])
        
        # Create destination directory (organized by class)
        class_dir = os.path.join(FINETUNE_DIR, row['species'])
        os.makedirs(class_dir, exist_ok=True)
        
        # Create destination path
        filename = Path(row['image_path']).name
        # Add source prefix to avoid filename collisions
        prefix = 'my_' if row['source'] == 'my_data' else 'ext_'
        dst_path = os.path.join(class_dir, f"{prefix}{filename}")
        
        # Copy file
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"  ⚠️  Failed to copy {src_path}: {e}")
    
    print(f"✅ Fine-tuning images copied to {FINETUNE_DIR}")
    
    print("\nCopying test images...")
    
    for idx, row in test_df.iterrows():
        src_path = os.path.join(BASE_DIR, row['image_path'])
        
        # Create destination directory (organized by class)
        class_dir = os.path.join(TEST_DIR, row['species'])
        os.makedirs(class_dir, exist_ok=True)
        
        # Create destination path
        filename = Path(row['image_path']).name
        dst_path = os.path.join(class_dir, filename)
        
        # Copy file
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"  ⚠️  Failed to copy {src_path}: {e}")
    
    print(f"✅ Test images copied to {TEST_DIR}")
    print()

else:
    print("Skipping image copying. Only CSV files created.")
    print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("DATASET CREATION COMPLETE")
print("=" * 80)
print()
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"\n📊 Fine-tuning Dataset:")
print(f"  - Labels: {finetune_csv_path}")
print(f"  - Total samples: {len(finetune_df)}")
print(f"  - Samples per class: ~200")
print(f"  - Images directory: {FINETUNE_DIR if copy_images in ['yes', 'y'] else 'Not copied'}")
print()
print(f"📊 Test Dataset:")
print(f"  - Labels: {test_csv_path}")
print(f"  - Total samples: {len(test_df)}")
print(f"  - Images directory: {TEST_DIR if copy_images in ['yes', 'y'] else 'Not copied'}")
print()
print(f"📄 Split Summary: {summary_path}")
print()
print("Next steps:")
print("  1. Verify the splits look correct")
print("  2. Use finetune_labels.csv for training your linear probe")
print("  3. Use test_labels.csv for final evaluation")
print("=" * 80)