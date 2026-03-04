
## code for running approach 1 (baseline) -> ResNet50 trained on Roboflow shark dataset

## prelims ----------------------------------------------------------------------------

## libraries
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU!")

## config
RESNET_CKPT = "./IEarth_CDT_shark_detection/best.pth"
LABEL_MAP = "./IEarth_CDT_shark_detection/label_map.json"

## inputs
LABELS_CSV = "./labels.csv"
BASE_DIR = "."
TARGET_CLASSES = ['whitetip_reef_shark', 'grey_reef_shark']

## outputs
OUTPUT_DIR = "./models/resnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MODEL SETUP
# ============================================================================

def get_model(arch, num_classes):
    if arch == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(pretrained=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m

def build_transform(img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

print("=" * 80)
print("ResNet50 Direct Classification Evaluation")
print("=" * 80)
print()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("Loading model checkpoint...")
ckpt = torch.load(RESNET_CKPT, map_location=device)
class_names = ckpt.get("class_names")

# Load from label map if available
reject_threshold = 0.4
if LABEL_MAP and os.path.exists(LABEL_MAP):
    with open(LABEL_MAP, 'r') as f:
        label_map = json.load(f)
        if "classes" in label_map:
            class_names = label_map["classes"]
        reject_threshold = float(label_map.get("reject_threshold", 0.4))

if not class_names:
    raise ValueError("No class names found in checkpoint or label_map.json")

print(f"Model classes: {class_names}")
print(f"Reject threshold: {reject_threshold}")
print()

arch = ckpt.get("arch", "resnet50")
model = get_model(arch, num_classes=len(class_names))
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

transform = build_transform(224)

# ============================================================================
# STEP 1: Load Labels and Filter to Target Classes
# ============================================================================
print("=" * 80)
print("STEP 1: Loading Ground Truth Labels")
print("=" * 80)

labels_df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(labels_df)} total labeled images")
print(f"Full class distribution:\n{labels_df['species'].value_counts()}\n")

# Filter to target classes only
labels_df = labels_df[labels_df['species'].isin(TARGET_CLASSES)].copy()
print(f"Filtered to {len(labels_df)} images for two-class evaluation")
print(f"Two-class distribution:\n{labels_df['species'].value_counts()}")
print()

# ============================================================================
# STEP 2: Classify Images
# ============================================================================
print("=" * 80)
print("STEP 2: Running ResNet50 Classification")
print("=" * 80)

results = []
failed = 0

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Classifying"):
    img_path = os.path.join(BASE_DIR, row['image_path'])
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found")
        failed += 1
        continue
    
    # Load and classify
    try:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        top_idx = int(probs.argmax())
        top_prob = float(probs[top_idx])
        
        # Apply reject threshold
        if top_prob < reject_threshold:
            pred_species = "background"
        else:
            pred_species = class_names[top_idx]
        
        results.append({
            'image_path': row['image_path'],
            'true_species': row['species'],
            'pred_species': pred_species,
            'confidence': top_prob
        })
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        failed += 1
        continue

print(f"\n✅ Classification complete!")
print(f"Successfully classified: {len(results)}")
print(f"Failed: {failed}")
print()

# Save predictions
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

# ============================================================================
# STEP 3: Compute Metrics
# ============================================================================
print("=" * 80)
print("STEP 3: Computing Evaluation Metrics")
print("=" * 80)

y_true = results_df['true_species'].values
y_pred = results_df['pred_species'].values
confidences = results_df['confidence'].values

# Classes
classes = sorted(TARGET_CLASSES)

# Check for predictions outside target classes
other_preds = results_df[~results_df['pred_species'].isin(TARGET_CLASSES)]
if len(other_preds) > 0:
    print(f"⚠️  Warning: {len(other_preds)} predictions ({len(other_preds)/len(results_df)*100:.1f}%) "
          f"were classified as other species:")
    print(other_preds['pred_species'].value_counts())
    print()

# Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=classes, zero_division=0
)

print("Per-Class Metrics:")
print("-" * 80)
metrics_df = pd.DataFrame({
    'Class': classes,
    'Support': support,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
print(metrics_df.to_string(index=False))
print()

# Macro and weighted averages
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=classes, average='macro', zero_division=0
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=classes, average='weighted', zero_division=0
)

print("Aggregate Metrics:")
print(f"  Macro Avg    - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
print(f"  Weighted Avg - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
print()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)
print("Confusion Matrix:")
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df)
print()

# Save metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'per_class_metrics.csv'), index=False)
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 
               'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
    'Score': [accuracy, macro_precision, macro_recall, macro_f1,
              weighted_precision, weighted_recall, weighted_f1]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)

# Classification Report
print("Detailed Classification Report:")
print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
print()

# ============================================================================
# STEP 4: Confidence Analysis
# ============================================================================
print("=" * 80)
print("STEP 4: Confidence Score Analysis")
print("=" * 80)

print(f"Overall Confidence Statistics:")
print(f"  Mean:   {confidences.mean():.4f}")
print(f"  Median: {np.median(confidences):.4f}")
print(f"  Std:    {confidences.std():.4f}")
print(f"  Min:    {confidences.min():.4f}")
print(f"  Max:    {confidences.max():.4f}")
print()

# Confidence by correctness
results_df['correct'] = results_df['true_species'] == results_df['pred_species']
correct_conf = results_df[results_df['correct']]['confidence']
incorrect_conf = results_df[~results_df['correct']]['confidence']

print(f"Correct Predictions   ({len(correct_conf):3d}): Mean confidence = {correct_conf.mean():.4f}")
print(f"Incorrect Predictions ({len(incorrect_conf):3d}): Mean confidence = {incorrect_conf.mean():.4f}")
print()

# Confidence by class
print("Confidence by True Class:")
for cls in classes:
    cls_conf = results_df[results_df['true_species'] == cls]['confidence']
    print(f"  {cls:25s}: Mean = {cls_conf.mean():.4f}, Std = {cls_conf.std():.4f}")
print()

# ============================================================================
# STEP 5: Error Analysis
# ============================================================================
print("=" * 80)
print("STEP 5: Error Analysis")
print("=" * 80)

errors = results_df[~results_df['correct']]
print(f"Total errors: {len(errors)} out of {len(results_df)} ({len(errors)/len(results_df)*100:.1f}%)")
print()

if len(errors) > 0:
    print("Error breakdown:")
    error_summary = errors.groupby(['true_species', 'pred_species']).size().reset_index(name='count')
    print(error_summary.to_string(index=False))
    print()
    
    # Save errors
    errors[['image_path', 'true_species', 'pred_species', 'confidence']].to_csv(
        os.path.join(OUTPUT_DIR, 'errors.csv'), index=False
    )
    print(f"✅ Saved error cases to errors.csv\n")

# ============================================================================
# STEP 6: Visualizations
# ============================================================================
print("=" * 80)
print("STEP 6: Creating Visualizations")
print("=" * 80)

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - ResNet50\n(n={len(results_df)})', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✅ Saved confusion_matrix.png")

# 2. Confidence Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[0].set_xlabel('Confidence Score', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Boxplot
box_data = [correct_conf, incorrect_conf]
bp = axes[1].boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
for box in bp['boxes']:
    box.set_alpha(0.6)
axes[1].set_ylabel('Confidence Score', fontsize=11)
axes[1].set_title('Confidence by Correctness', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# Confidence by class
class_conf_data = [results_df[results_df['true_species'] == cls]['confidence'] 
                   for cls in classes]
bp2 = axes[2].boxplot(class_conf_data, labels=[c.replace('_', '\n') for c in classes], patch_artist=True)
for box in bp2['boxes']:
    box.set_facecolor('skyblue')
    box.set_alpha(0.6)
axes[2].set_ylabel('Confidence Score', fontsize=11)
axes[2].set_title('Confidence by True Class', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
print("✅ Saved confidence_analysis.png")

# 3. Per-class Performance
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(classes))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2E86AB')
bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#A23B72')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#F18F01')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Species', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in classes])
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
print("✅ Saved per_class_metrics.png")

print()
print("=" * 80)
print("EVALUATION COMPLETE - ResNet50 (Direct Classification)")
print("=" * 80)
print(f"\n📊 SUMMARY:")
print(f"  Test samples:     {len(results_df)}")
print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro F1:         {macro_f1:.4f}")
print(f"  Weighted F1:      {weighted_f1:.4f}")
print(f"\n📁 Results saved to: {OUTPUT_DIR}")
print(f"  - predictions.csv")
print(f"  - summary_metrics.csv")
print(f"  - per_class_metrics.csv")
print(f"  - errors.csv")
print(f"  - confusion_matrix.png")
print(f"  - confidence_analysis.png")
print(f"  - per_class_metrics.png")
print("=" * 80)
