## Code to retry failed shark crops from 01_crop_sharks.py
## Input: Original CSV and existing cropped images
## Output: Previously failed crops in a new folder

import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm import tqdm

## set paths
CSV_PATH = "./sharktrack_results/internal_results/output.csv"
VIDEO_DIR = "/gws/nopw/j04/iecdt/shark_bruvs" # update with where your videos are stored
EXISTING_METADATA = "./cropped_sharks/cropped_metadata.csv"
OUTPUT_DIR = "./cropped_sharks2"

# Increase read attempts even more for problematic videos
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '50000'

def crop_shark_from_video(video_path, frame_num, xmin, ymin, xmax, ymax):
    """
    Extract a specific frame from video and crop to bounding box
    """
    cap = cv2.VideoCapture(video_path)
    
    # try setting backend explicitly for problematic videos
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # crop using bounding box
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    
    # ensuring coordinates are within frame bounds
    h, w = frame.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    cropped = frame[ymin:ymax, xmin:xmax]
    
    return cropped

def main():
    # creating output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # loading original CSV with all detections
    print("Loading original SharkTrack results...")
    df_original = pd.read_csv(CSV_PATH)
    
    # loading existing successful crops
    print("Loading existing crop metadata...")
    df_existing = pd.read_csv(EXISTING_METADATA)
    
    # creating set of successfully cropped identifiers
    existing_ids = set()
    for _, row in df_existing.iterrows():
        # extract identifying info from filename or use metadata columns
        identifier = f"{row['video_name']}_{row['track_id']}_{row['frame']}"
        existing_ids.add(identifier)
    
    print(f"Found {len(existing_ids)} existing successful crops")
    print(f"Original detections: {len(df_original)}")
    
    # find failed crops
    failed_crops = []
    for idx, row in df_original.iterrows():
        identifier = f"{row['video_name']}_{row['track_id']}_{row['frame']}"
        if identifier not in existing_ids:
            failed_crops.append(row)
    
    df_failed = pd.DataFrame(failed_crops)
    print(f"Found {len(df_failed)} failed crops to retry")
    
    if len(df_failed) == 0:
        print("No failed crops to retry!")
        return
    
    # retry cropping failed detections
    cropped_images = []
    
    for idx, row in tqdm(df_failed.iterrows(), total=len(df_failed), desc="Retrying failed crops"):
        video_name = row['video_name']
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # Extract and crop
        cropped = crop_shark_from_video(
            video_path,
            frame_num=row['frame'],
            xmin=row['xmin'],
            ymin=row['ymin'],
            xmax=row['xmax'],
            ymax=row['ymax']
        )
        
        if cropped is None or cropped.size == 0:
            continue
        
        # Save with same naming convention
        output_filename = f"video_{video_name.split('.')[0]}_track_{row['track_id']}_frame_{row['frame']}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        cv2.imwrite(output_path, cropped)
        
        cropped_images.append({
            'image_path': output_path,
            'video_name': video_name,
            'track_id': row['track_id'],
            'frame': row['frame'],
            'time': row['time'],
            'confidence': row['confidence']
        })
    
    # saving metadata
    cropped_df = pd.DataFrame(cropped_images)
    metadata_path = os.path.join(OUTPUT_DIR, "cropped_metadata.csv")
    cropped_df.to_csv(metadata_path, index=False)
    
    print(f"\n✅ Successfully cropped {len(cropped_images)} previously failed images")
    print(f"❌ Still failed: {len(df_failed) - len(cropped_images)}")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print(f"📄 Metadata: {metadata_path}")

if __name__ == "__main__":
    main()