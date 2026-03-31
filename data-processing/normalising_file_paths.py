import pandas as pd
import re

df = pd.read_csv('./data/finetune_labels_old.csv')

def normalise_path(path):
    # match giga_cropped paths
    m = re.search(r'(giga_cropped/.+)', path)
    if m:
        return './data/' + m.group(1)
    
    # match cropped_sharks paths
    m = re.search(r'(cropped_sharks/.+)', path)
    if m:
        return './data/' + m.group(1)
    
    # match roboflow2 paths
    m = re.search(r'(roboflow2/.+)', path)
    if m:
        return './data/' + m.group(1)
    
    return path  # leave unchanged if no match

df['image_path'] = df['image_path'].apply(normalise_path)

df.to_csv('./data/finetune_labels.csv', index=False)
print("Done. Sample output:")
print(df['image_path'].head(10).to_string())