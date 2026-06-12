
## code for preprocessing BRUVS videos using SharkTrack
## input: SharkTrack output 
## output: a folder of cropped images of shark stills from the videos, and a .csv file with the corresponding metadata (filename, date, time, location, etc.)

## libraries
import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm import tqdm

## set paths
CSV_PATH = "./iecdt-shark-id/sharktrack_results/internal_results/output.csv"
VIDEO_DIR = "/gws/nopw/j04/iecdt/shark_bruvs" 
OUTPUT_DIR = "./data/cropped_sharks"

# setting higher read attempts for complex video formats
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

## test mode
TEST_ROWS = None

## function
def crop_shark_from_video(video_path, frame_num, xmin, ymin, xmax, ymax):
    """
    Extract a specific frame from video and crop to bounding box
    """
    cap = cv2.VideoCapture(video_path)
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
    # creating output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # reading CSV
    print("Loading SharkTrack results...")
    df = pd.read_csv(CSV_PATH)
    if TEST_ROWS is not None:
        print(f"🧪 TEST MODE: Processing only first {TEST_ROWS} rows")
        df = df.head(TEST_ROWS)
    
    print(f"Found {len(df)} detections across {df['track_id'].nunique()} unique tracks")
    
    # processing each detection
    cropped_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping sharks"):
        video_name = row['video_name']
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # extracting and cropping
        cropped = crop_shark_from_video(
            video_path,
            frame_num=row['frame'],
            xmin=row['xmin'],
            ymin=row['ymin'],
            xmax=row['xmax'],
            ymax=row['ymax']
        )
        
        if cropped is None or cropped.size == 0:
            print(f"Warning: Failed to crop detection {idx}")
            continue
        
        # saving
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
    
    print(f"\n✅ Cropped {len(cropped_images)} shark images")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print(f"📄 Metadata: {metadata_path}")

if __name__ == "__main__":
    main()
