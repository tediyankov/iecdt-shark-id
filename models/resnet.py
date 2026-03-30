# ResNet50 Evaluation on Balanced Test Set
# job: 

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

# ============================================================================
# config 
# ============================================================================
RESNET_CKPT = "./IEarth_CDT_shark_detection/best.pth"
LABEL_MAP = "./IEarth_CDT_shark_detection/label_map.json"

# our data
TEST_LABELS_CSV = "./data/labels.csv"
BASE_DIR = "."

# output
OUTPUT_DIR = "./results/resnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# class mapping
# ============================================================================

# mapping BRUVS labels to ResNet classes
CLASS_MAPPING = {
    'whitetip_reef_shark': 'Whitetip_Shark',
    'blacktip_reef_shark': 'Blacktip_Shark',
    'tawny_nurse_shark': 'Nurse_Shark',
    # grey_reef_shark has no match - we exclude it
    # unclear/other - will be excluded (not a shark species)
}

# classes that can be evaluated (have a mapping)
EVALUABLE_CLASSES = list(CLASS_MAPPING.keys())

# ============================================================================
# setup of model
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
print("ResNet50 Evaluation on Balanced Test Set")
print("=" * 80)
print()

# loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("Loading model checkpoint...")
ckpt = torch.load(RESNET_CKPT, map_location=device)
class_names = ckpt.get("class_names")

# loading from label map if available
reject_threshold = 0.4
if LABEL_MAP and os.path.exists(LABEL_MAP):
    with open(LABEL_MAP, 'r') as f:
        label_map = json.load(f)
        if "classes" in label_map:
            class_names = label_map["classes"]
        reject_threshold = float(label_map.get("reject_threshold", 0.4))

if not class_names:
    raise ValueError("No class names found in checkpoint or label_map.json")

print(f"ResNet classes: {class_names}")
print(f"Reject threshold: {reject_threshold}")
print()

print("Class Mapping:")
for bruvs_class, resnet_class in CLASS_MAPPING.items():
    print(f"  {bruvs_class:25s} → {resnet_class}")
print(f"grey_reef_shark → [EXCLUDED - no match]")
print(f"unclear/other → [EXCLUDED - not a species]")
print()

arch = ckpt.get("arch", "resnet50")
model = get_model(arch, num_classes=len(class_names))
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

transform = build_transform(224)

# ============================================================================
# loading test set and filter to evaluate classes
# ============================================================================
print("=" * 80)
print("STEP 1: loading test set")
print("=" * 80)

test_df = pd.read_csv(TEST_LABELS_CSV)
print(f"Loaded {len(test_df)} total test images")
print(f"Full test set distribution:\n{test_df['species'].value_counts()}\n")

# filtering to only classes that can be mapped
test_df_orig = test_df.copy()
test_df = test_df[test_df['species'].isin(EVALUABLE_CLASSES)].copy()

excluded_count = len(test_df_orig) - len(test_df)
excluded_grey = len(test_df_orig[test_df_orig['species'] == 'grey_reef_shark'])
excluded_unclear = len(test_df_orig[test_df_orig['species'] == 'unclear/other'])

print(f"Excluded {excluded_count} images:")
print(f"- grey_reef_shark: {excluded_grey} (no ResNet match)")
print(f"- unclear/other: {excluded_unclear} (not a species)")
print()
print(f"Evaluating on {len(test_df)} images:")
print(test_df['species'].value_counts())
print()

if len(test_df) == 0:
    print("ERROR: No evaluable samples after filtering!")
    exit(1)

# ============================================================================
# classifying images
# ============================================================================
print("=" * 80)
print("STEP 2: running ResNet50 classification")
print("=" * 80)

results = []
failed = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
    img_path = os.path.join(BASE_DIR, row['image_path'])
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found")
        failed += 1
        continue
    
    # loading and classifying
    try:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        top_idx = int(probs.argmax())
        top_prob = float(probs[top_idx])
        
        # applying reject threshold
        if top_prob < reject_threshold:
            pred_species = "background"
        else:
            pred_species = class_names[top_idx]
        
        results.append({
            'image_path': row['image_path'],
            'true_species_original': row['species'],
            'true_species_mapped': CLASS_MAPPING[row['species']],
            'pred_species': pred_species,
            'confidence': top_prob
        })
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        failed += 1
        continue

print(f"\n✅ classification complete woohoo!")
print(f"Successfully classified: {len(results)}")
print(f"Failed: {failed}")
print()

# saving preds
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

# ============================================================================
# computing metrics using mapped classes
# ============================================================================
print("=" * 80)
print("STEP 3: computing eval metrics")
print("=" * 80)

# using mapped true labels for comparison
y_true = results_df['true_species_mapped'].values
y_pred = results_df['pred_species'].values
confidences = results_df['confidence'].values

# getting unique mapped classes
mapped_classes = sorted(CLASS_MAPPING.values())

# overall accuracy (exact match)
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

# pred distribution
print("Prediction distribution:")
pred_counts = results_df['pred_species'].value_counts()
print(pred_counts)
print()

# checking how many predictions match any of our target classes
correct_class = results_df['pred_species'].isin(mapped_classes)
print(f"Predictions in target classes: {correct_class.sum()} / {len(results_df)} ({correct_class.sum()/len(results_df)*100:.1f}%)")
print()

# per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=mapped_classes, zero_division=0
)

print("Per-Class Metrics:")
print("-" * 80)

# creating reverse mapping for display
reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
metrics_df = pd.DataFrame({
    'ResNet Class': mapped_classes,
    'BRUVS Class': [reverse_mapping[c] for c in mapped_classes],
    'Support': support,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
print(metrics_df.to_string(index=False))
print()

# macro and weighted averages
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=mapped_classes, average='macro', zero_division=0
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=mapped_classes, average='weighted', zero_division=0
)

print("Aggregate Metrics:")
print(f"Macro Avg - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
print(f"Weighted Avg - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
print()

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=mapped_classes)
print("Confusion Matrix:")
cm_df = pd.DataFrame(cm, index=mapped_classes, columns=mapped_classes)
print(cm_df)
print()

# saving metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'per_class_metrics.csv'), index=False)
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 
               'Weighted Precision', 'Weighted Recall', 'Weighted F1',
               'Total Test Samples', 'Excluded (grey_reef)', 'Excluded (unclear/other)'],
    'Score': [accuracy, macro_precision, macro_recall, macro_f1,
              weighted_precision, weighted_recall, weighted_f1,
              len(results_df), excluded_grey, excluded_unclear]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)

# classification report
print("Detailed classification report:")
print(classification_report(y_true, y_pred, labels=mapped_classes, zero_division=0))
print()

# ============================================================================
# confidence analysis
# ============================================================================
print("=" * 80)
print("STEP 4: Confidence Score Analysis")
print("=" * 80)

print(f"Overall confidence stats:")
print(f"Mean: {confidences.mean():.4f}")
print(f"Median: {np.median(confidences):.4f}")
print(f"Std: {confidences.std():.4f}")
print(f"Min: {confidences.min():.4f}")
print(f"Max: {confidences.max():.4f}")
print()

# conf by correctness
results_df['correct'] = results_df['true_species_mapped'] == results_df['pred_species']
correct_conf = results_df[results_df['correct']]['confidence']
incorrect_conf = results_df[~results_df['correct']]['confidence']

if len(correct_conf) > 0:
    print(f"Correct Predictions ({len(correct_conf):4d}): Mean confidence = {correct_conf.mean():.4f}")
else:
    print(f"Correct Predictions (0): No correct predictions")
    
if len(incorrect_conf) > 0:
    print(f"Incorrect Predictions ({len(incorrect_conf):4d}): Mean confidence = {incorrect_conf.mean():.4f}")
print()

# Confidence by class
print("Confidence by True Class (BRUVS labels):")
for bruvs_cls in EVALUABLE_CLASSES:
    cls_conf = results_df[results_df['true_species_original'] == bruvs_cls]['confidence']
    if len(cls_conf) > 0:
        print(f"  {bruvs_cls:25s}: Mean = {cls_conf.mean():.4f}, Std = {cls_conf.std():.4f}, n={len(cls_conf)}")
print()

# ============================================================================
# error analysis
# ============================================================================
print("=" * 80)
print("STEP 5: Error Analysis")
print("=" * 80)

errors = results_df[~results_df['correct']]
print(f"Total errors: {len(errors)} out of {len(results_df)} ({len(errors)/len(results_df)*100:.1f}%)")
print()

if len(errors) > 0:
    print("Error breakdown (showing BRUVS labels):")
    error_summary = errors.groupby(['true_species_original', 'pred_species']).size().reset_index(name='count')
    error_summary_sorted = error_summary.sort_values('count', ascending=False)
    print(error_summary_sorted.to_string(index=False))
    print()
    
    # saving errors
    errors[['image_path', 'true_species_original', 'true_species_mapped', 'pred_species', 'confidence']].to_csv(
        os.path.join(OUTPUT_DIR, 'errors.csv'), index=False
    )
    print(f"✅ Saved error cases to errors.csv\n")

# ============================================================================
# visualisations
# ============================================================================
print("=" * 80)
print("STEP 6: Creating Visualizations")
print("=" * 80)

# confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=mapped_classes, yticklabels=mapped_classes,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - ResNet50 (Mapped Classes)\n(n={len(results_df)}, excluded={excluded_count})', 
          fontsize=12, fontweight='bold')
plt.ylabel('True Label (ResNet classes)', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✅ Saved confusion_matrix.png")

# confidence analysis
if len(correct_conf) > 0 and len(incorrect_conf) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # histogram
    axes[0].hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # boxplot - correctness
    box_data = [correct_conf, incorrect_conf]
    bp = axes[1].boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    axes[1].set_ylabel('Confidence Score', fontsize=11)
    axes[1].set_title('Confidence by Correctness', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # boxplot - by class
    class_conf_data = [results_df[results_df['true_species_original'] == cls]['confidence'] 
                       for cls in EVALUABLE_CLASSES]
    bp2 = axes[2].boxplot(class_conf_data, 
                          labels=[c.replace('_reef_shark', '').replace('_', ' ').title() for c in EVALUABLE_CLASSES],
                          patch_artist=True)
    for box in bp2['boxes']:
        box.set_facecolor('skyblue')
        box.set_alpha(0.6)
    axes[2].set_ylabel('Confidence Score', fontsize=11)
    axes[2].set_title('Confidence by True Class', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved confidence_analysis.png")
else:
    print("⚠️  Skipped confidence_analysis.png (no correct predictions or no errors)")

# per-class performance
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(mapped_classes))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2E86AB')
bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#A23B72')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#F18F01')

# adding value labels and support counts
for i, (bars, metric_name) in enumerate([(bars1, 'Precision'), (bars2, 'Recall'), (bars3, 'F1')]):
    for bar, supp in zip(bars, support):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Species (ResNet Classes)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ') for c in mapped_classes], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# adding support counts below x-axis
for i, (cls, supp) in enumerate(zip(mapped_classes, support)):
    ax.text(i, -0.15, f'n={int(supp)}', ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
print("✅ Saved per_class_metrics.png")

print()
print("=" * 80)
print("EVALUATION COMPLETE - ResNet50 (Balanced Test Set)")
print("=" * 80)
print(f"\n📊 SUMMARY:")
print(f"  Test samples:     {len(results_df)}")
print(f"  Excluded:         {excluded_count} ({excluded_grey} grey_reef, {excluded_unclear} unclear)")
print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro F1:         {macro_f1:.4f}")
print(f"  Weighted F1:      {weighted_f1:.4f}")
print(f"\n📁 Results saved to: {OUTPUT_DIR}")
print(f"  - predictions.csv")
print(f"  - summary_metrics.csv")
print(f"  - per_class_metrics.csv")
print(f"  - errors.csv")
print(f"  - confusion_matrix.png")
print(f"  - confidence_analysis.png (if applicable)")
print(f"  - per_class_metrics.png")
print("=" * 80)