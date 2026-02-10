# iecdt-shark-id

Temporary instructions to run the labeller: 

1. Clone this repo on your local machine (not JASMIN)
2. Download your assigned folder of unlabelled images 

   Greg: your images are here: https://drive.google.com/drive/folders/1DpYr1I02CzbG4H8aqgRcGOdSWM7o9UZl?usp=sharing

   Darina: your images are here: https://drive.google.com/drive/folders/1Bf5S6grnDIaBPj9zcYD8qV5_3o7gjCg7?usp=sharing
   
4. Name your assigned folder 'cropped_sharks' and drag it into your current directory (ie the cloned repo)
5. Navigate to the image labelling script in `code/02_image_labelling.py`
6. Make sure to update `IMAGE_DIR` to match the path of YOUR cropped_sharks (ie the one you downloaded and renamed from the Drive link) and `OUTPUT_CSV` to where you wish your labelled dataset to live.
7. Run the script with `python code/02_image_labelling.py`. This will open the first image in your folder.
8. All you have to do to label the image currently on your screen is press one of the following keys: 

```
=== Shark Species Labeling ===
Press:
  1: grey_reef_shark
  2: blacktip_reef_shark
  3: whitetip_reef_shark
  4: tawny_nurse_shark
  0: unclear/other
  q: quit
  s: skip
  b: go back
  ESC: save and quit
========================================
```

8. Once you're done, send your CSV to teodor.yankov@new.ox.ac.uk. Thanks!! 
