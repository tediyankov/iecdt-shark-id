import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, top_k_accuracy_score
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data paths
TEST_CSV = "./data/test_labels.csv"
BASE_DIR = "."
EXTERNAL_BASE_DIR = "/gws/nopw/j04/iecdt/shark_bruvs/roboflow2" # update this if using an external labelled image database

# Model path
MODEL_DIR = "./results/dinov2"
MODEL_PATH = os.path.join(MODEL_DIR, "best_linear_probe.pth")

# Output
OUTPUT_DIR = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/dinov2_evaluation2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target classes
TARGET_CLASSES = ['grey_reef_shark', 'blacktip_reef_shark', 'whitetip_reef_shark', 'tawny_nurse_shark']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

print("=" * 80)
print("DINOv2 Evaluation Only (Loading Saved Model)")
print("=" * 80)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# DATASET CLASS
# ============================================================================
class SharkDataset(Dataset):
    def __init__(self, csv_path, base_dir, external_base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.external_base_dir = external_base_dir
        self.transform = transform
        
        # Filter to target classes only
        self.df = self.df[self.df['species'].isin(TARGET_CLASSES)].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Determine image path based on source
        if 'source' in row and row['source'] == 'external':
            img_path = os.path.join(self.external_base_dir, row['image_path'])
        else:
            img_path = os.path.join(self.base_dir, row['image_path'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = CLASS_TO_IDX[row['species']]
        
        return image, label, row['image_path']

# ============================================================================
# DATA TRANSFORMS
# ============================================================================
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================================================
# LOAD TEST DATASET
# ============================================================================
print("=" * 80)
print("Loading Test Dataset")
print("=" * 80)

test_dataset = SharkDataset(TEST_CSV, BASE_DIR, EXTERNAL_BASE_DIR, transform=test_transform)
print(f"Test dataset:  {len(test_dataset)} images")

test_labels = [test_dataset.df.iloc[i]['species'] for i in range(len(test_dataset))]
print("\nTest distribution:")
print(pd.Series(test_labels).value_counts())
print()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================================
# LOAD DINOV2 MODEL
# ============================================================================
print("=" * 80)
print("Loading DINOv2 Backbone")
print("=" * 80)

print("Loading DINOv2 ViT-B/14...")
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14 = dinov2_vitb14.to(DEVICE)

for param in dinov2_vitb14.parameters():
    param.requires_grad = False

dinov2_vitb14.eval()
print("✅ DINOv2 loaded")

# Get embedding dimension
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_output = dinov2_vitb14(dummy_input)
    embedding_dim = dummy_output.shape[1]

print(f"Embedding dimension: {embedding_dim}")
print()

# ============================================================================
# LINEAR PROBE
# ============================================================================
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

linear_probe = LinearProbe(embedding_dim, len(TARGET_CLASSES)).to(DEVICE)

# Load trained weights
print(f"Loading trained linear probe from: {MODEL_PATH}")
linear_probe.load_state_dict(torch.load(MODEL_PATH))
linear_probe.eval()
print("✅ Model loaded")
print()

# ============================================================================
# EVALUATION
# ============================================================================
print("=" * 80)
print("Evaluating on Test Set")
print("=" * 80)
print()

all_labels = []
all_preds = []
all_probs = []
all_features = []
all_image_paths = []

with torch.no_grad():
    for images, labels, image_paths in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Extract features
        features = dinov2_vitb14(images)
        
        # Classify
        outputs = linear_probe(features)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_features.extend(features.cpu().numpy())
        all_image_paths.extend(image_paths)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_features = np.array(all_features)

# ============================================================================
# COMPUTE METRICS
# ============================================================================
print("=" * 80)
print("Computing Metrics")
print("=" * 80)
print()

# Top-1 Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Top-3 Accuracy
top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(len(TARGET_CLASSES)))
print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
print()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=range(len(TARGET_CLASSES)), zero_division=0
)

print("Per-Class Metrics:")
print("-" * 80)
metrics_df = pd.DataFrame({
    'Class': TARGET_CLASSES,
    'Support': support,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
print(metrics_df.to_string(index=False))
print()

# Aggregate metrics
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro', zero_division=0
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted', zero_division=0
)

print("Aggregate Metrics:")
print(f"  Macro Avg    - Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}")
print(f"  Weighted Avg - Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f}, F1: {weighted_f1:.4f}")
print()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=range(len(TARGET_CLASSES)))
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES))
print()

# Classification Report
print("Detailed Classification Report:")
print(classification_report(all_labels, all_preds, 
                           target_names=TARGET_CLASSES, zero_division=0))
print()

# Save metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'per_class_metrics.csv'), index=False)
summary_df = pd.DataFrame({
    'Metric': ['Top-1 Accuracy', 'Top-3 Accuracy', 
               'Macro Precision', 'Macro Recall', 'Macro F1',
               'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
    'Score': [accuracy, top3_acc, macro_p, macro_r, macro_f1,
              weighted_p, weighted_r, weighted_f1]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)

# Save predictions
predictions_df = pd.DataFrame({
    'image_path': all_image_paths,
    'true_label': [IDX_TO_CLASS[i] for i in all_labels],
    'pred_label': [IDX_TO_CLASS[i] for i in all_preds],
    'confidence': np.max(all_probs, axis=1),
    **{f'prob_{cls}': all_probs[:, idx] for idx, cls in enumerate(TARGET_CLASSES)}
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

# ============================================================================
# CONFIDENCE SCORE ANALYSIS
# ============================================================================
print("=" * 80)
print("Confidence Score Analysis")
print("=" * 80)
print()

confidences = np.max(all_probs, axis=1)
correct_mask = all_labels == all_preds

correct_conf = confidences[correct_mask]
incorrect_conf = confidences[~correct_mask]

print(f"Overall Confidence Statistics:")
print(f"  Mean:   {confidences.mean():.4f}")
print(f"  Median: {np.median(confidences):.4f}")
print(f"  Std:    {confidences.std():.4f}")
print()

print(f"Correct predictions   ({len(correct_conf):4d}): Mean = {correct_conf.mean():.4f}")
print(f"Incorrect predictions ({len(incorrect_conf):4d}): Mean = {incorrect_conf.mean():.4f}")
print()

# ============================================================================
# FEATURE QUALITY ANALYSIS
# ============================================================================
print("=" * 80)
print("Feature Quality Analysis (Intra-class vs Inter-class Distance)")
print("=" * 80)
print()

from scipy.spatial.distance import cdist

# Compute intra-class and inter-class distances
intra_class_dists = []
inter_class_dists = []

for cls_idx in range(len(TARGET_CLASSES)):
    cls_features = all_features[all_labels == cls_idx]
    
    # Intra-class: average pairwise distance within class
    if len(cls_features) > 1:
        intra_dists = cdist(cls_features, cls_features, metric='euclidean')
        # Get upper triangle (excluding diagonal)
        intra_dists = intra_dists[np.triu_indices_from(intra_dists, k=1)]
        intra_class_dists.extend(intra_dists)
    
    # Inter-class: distance to other classes
    for other_cls_idx in range(len(TARGET_CLASSES)):
        if other_cls_idx != cls_idx:
            other_features = all_features[all_labels == other_cls_idx]
            inter_dists = cdist(cls_features, other_features, metric='euclidean')
            inter_class_dists.extend(inter_dists.flatten())

intra_class_dists = np.array(intra_class_dists)
inter_class_dists = np.array(inter_class_dists)

print(f"Intra-class distance (within species):")
print(f"  Mean: {intra_class_dists.mean():.4f}")
print(f"  Std:  {intra_class_dists.std():.4f}")
print()

print(f"Inter-class distance (between species):")
print(f"  Mean: {inter_class_dists.mean():.4f}")
print(f"  Std:  {inter_class_dists.std():.4f}")
print()

separation_ratio = inter_class_dists.mean() / intra_class_dists.mean()
print(f"Separation Ratio (inter/intra): {separation_ratio:.4f}")
print(f"  (Higher is better - want inter > intra)")
print()

# ============================================================================
# t-SNE VISUALIZATION (FIXED)
# ============================================================================
print("=" * 80)
print("Creating t-SNE Visualization")
print("=" * 80)
print()

print("Computing t-SNE embeddings (this may take a few minutes)...")
# Subsample if too many points
max_samples = 2000
if len(all_features) > max_samples:
    indices = np.random.choice(len(all_features), max_samples, replace=False)
    tsne_features = all_features[indices]
    tsne_labels = all_labels[indices]
    tsne_preds = all_preds[indices]
    print(f"  Subsampled to {max_samples} points for visualization")
else:
    tsne_features = all_features
    tsne_labels = all_labels
    tsne_preds = all_preds

# FIXED: Use max_iter instead of n_iter
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_2d = tsne.fit_transform(tsne_features)

print("✅ t-SNE complete")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("=" * 80)
print("Creating Visualizations")
print("=" * 80)
print()

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[c.replace('_', ' ').title() for c in TARGET_CLASSES],
            yticklabels=[c.replace('_', ' ').title() for c in TARGET_CLASSES],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - DINOv2 Linear Probe\n(n={len(all_labels)})',
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✅ Saved confusion_matrix.png")

# 2. Per-class metrics
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(TARGET_CLASSES))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2E86AB')
bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#A23B72')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#F18F01')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Species', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance - DINOv2', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in TARGET_CLASSES], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
print("✅ Saved per_class_metrics.png")

# 3. Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[0].set_xlabel('Confidence Score', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

box_data = [correct_conf, incorrect_conf]
bp = axes[1].boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
for box in bp['boxes']:
    box.set_alpha(0.6)
axes[1].set_ylabel('Confidence Score', fontsize=11)
axes[1].set_title('Confidence by Correctness', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
print("✅ Saved confidence_analysis.png")

# 4. Feature quality
plt.figure(figsize=(10, 6))
plt.hist(intra_class_dists, bins=50, alpha=0.6, label='Intra-class', color='blue', edgecolor='black')
plt.hist(inter_class_dists, bins=50, alpha=0.6, label='Inter-class', color='orange', edgecolor='black')
plt.xlabel('Euclidean Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Feature Space Distance Distribution\nSeparation Ratio: {separation_ratio:.2f}',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_quality.png'), dpi=300, bbox_inches='tight')
print("✅ Saved feature_quality.png")

# 5. t-SNE visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Color by true label
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for idx, cls in enumerate(TARGET_CLASSES):
    mask = tsne_labels == idx
    axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors[idx], label=cls.replace('_', ' ').title(),
                   alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
axes[0].set_title('t-SNE Colored by True Label', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlabel('t-SNE 1', fontsize=11)
axes[0].set_ylabel('t-SNE 2', fontsize=11)

# Color by prediction (correct vs incorrect)
correct_mask_tsne = tsne_labels == tsne_preds
axes[1].scatter(embeddings_2d[correct_mask_tsne, 0], embeddings_2d[correct_mask_tsne, 1],
               c='green', label='Correct', alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
axes[1].scatter(embeddings_2d[~correct_mask_tsne, 0], embeddings_2d[~correct_mask_tsne, 1],
               c='red', label='Incorrect', alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
axes[1].set_title('t-SNE Colored by Prediction Correctness', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xlabel('t-SNE 1', fontsize=11)
axes[1].set_ylabel('t-SNE 2', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
print("✅ Saved tsne_visualization.png")

print()
print("=" * 80)
print("EVALUATION COMPLETE - DINOv2 Linear Probe")
print("=" * 80)
print()
print(f"📊 SUMMARY:")
print(f"  Test samples:      {len(all_labels)}")
print(f"  Top-1 Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Top-3 Accuracy:    {top3_acc:.4f} ({top3_acc*100:.2f}%)")
print(f"  Macro F1:          {macro_f1:.4f}")
print(f"  Weighted F1:       {weighted_f1:.4f}")
print(f"  Separation Ratio:  {separation_ratio:.4f}")
print()
print(f"📁 Results saved to: {OUTPUT_DIR}")
print("=" * 80)