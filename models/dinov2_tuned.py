
## code to hyperparameter tune the DinoV2 model

## PRELIMS ------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import json
import copy

## CONFIG ------------------

FINETUNE_CSV = "./data/finetune_labels.csv"
TEST_CSV = "./data/test_labels.csv"
BASE_DIR = "."
EXTERNAL_BASE_DIR = "/gws/nopw/j04/iecdt/shark_bruvs/roboflow2" 

OUTPUT_DIR = "./results/dinov2_tuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRETRAINED_MODEL = "/gws/nopw/j04/iecdt/tyankov/iecdt-shark-id/dinov2_evaluation/best_linear_probe.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.2

TARGET_CLASSES = ['grey_reef_shark', 'blacktip_reef_shark', 'whitetip_reef_shark', 'tawny_nurse_shark']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

## HYPERPARAMETER TUNING SET UP ------------------

HYPERPARAM_GRID = {
    'learning_rate': [0.0001, 0.0005, 0.001], 
    'batch_size': [32, 64],  
    'hidden_dim': [256, 512],  
    'dropout': [0.2, 0.3, 0.5],
    'optimizer': ['adam', 'adamw'],
    'weight_decay': [0.0, 0.0001, 0.001],  
}

print("=" * 80)
print("DINOv2 Warm-Start Grid Search")
print("=" * 80)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()
print(f"Warm-starting from: {PRETRAINED_MODEL}")
print()
print("Hyperparameter Grid:")
for param, values in HYPERPARAM_GRID.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in HYPERPARAM_GRID.values()])
print(f"\nTotal combinations: {total_combinations}")
print()

## CUSTOM DATASET CLASS ------------------

class SharkDataset(Dataset):
    def __init__(self, csv_path, base_dir, external_base_dir, transform=None, indices=None):
        df = pd.read_csv(csv_path)
        df = df[df['species'].isin(TARGET_CLASSES)].reset_index(drop=True)
        
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
        else:
            self.df = df
            
        self.base_dir = base_dir
        self.external_base_dir = external_base_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if 'source' in row and row['source'] == 'external':
            img_path = os.path.join(self.external_base_dir, row['image_path'])
        else:
            img_path = os.path.join(self.base_dir, row['image_path'])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = CLASS_TO_IDX[row['species']]
        return image, label, row['image_path']

## DATA TRANSFORMS ------------------

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

## LOADING AND SPLITTING DATA ------------------

print("=" * 80)
print("Loading and Splitting Data")
print("=" * 80)

full_train_dataset = SharkDataset(FINETUNE_CSV, BASE_DIR, EXTERNAL_BASE_DIR, transform=train_transform)

labels = [full_train_dataset.df.iloc[i]['species'] for i in range(len(full_train_dataset))]
train_idx, val_idx = train_test_split(
    range(len(full_train_dataset)), 
    test_size=VAL_SPLIT, 
    stratify=labels,
    random_state=42
)

print(f"Total training data: {len(full_train_dataset)}")
print(f"Train split: {len(train_idx)} samples")
print(f"Val split: {len(val_idx)} samples")
print()

train_dataset = SharkDataset(FINETUNE_CSV, BASE_DIR, EXTERNAL_BASE_DIR, 
                             transform=train_transform, indices=train_idx)
val_dataset = SharkDataset(FINETUNE_CSV, BASE_DIR, EXTERNAL_BASE_DIR, 
                           transform=val_transform, indices=val_idx)

test_dataset = SharkDataset(TEST_CSV, BASE_DIR, EXTERNAL_BASE_DIR, transform=val_transform)
print(f"Test dataset: {len(test_dataset)} samples")
print()

## LOADING DINOV2 BACKBONE ------------------

print("=" * 80)
print("Loading DINOv2 Backbone")
print("=" * 80)

print("Loading DINOv2 ViT-B/14...")
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14 = dinov2_vitb14.to(DEVICE)

for param in dinov2_vitb14.parameters():
    param.requires_grad = False

dinov2_vitb14.eval()
print("✅ DINOv2 loaded and frozen")

with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_output = dinov2_vitb14(dummy_input)
    embedding_dim = dummy_output.shape[1]

print(f"Embedding dimension: {embedding_dim}")
print()

## LOADING PRETRAINED WEIGHTS ------------------

print("=" * 80)
print("Loading Pretrained Weights")
print("=" * 80)

pretrained_state_dict = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
print(f"✅ Loaded pretrained weights from {PRETRAINED_MODEL}")
print()

## CUSTOM CLASS FOR LINEAR PROBE ------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

## TRAINING FUNCTION WITH WARM START ------------------

def train_model(hyperparams, train_loader, val_loader, pretrained_weights, 
                max_epochs=30, patience=7):
    
    # creating model
    model = LinearProbe(
        embedding_dim, 
        len(TARGET_CLASSES),
        hidden_dim=hyperparams['hidden_dim'],
        dropout=hyperparams['dropout']
    ).to(DEVICE)
    
    print("→ Training from scratch with well-tuned DINOv2 features")
    
    # optimiser
    if hyperparams['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(max_epochs):
        # training
        model.train()
        train_loss = 0.0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.no_grad():
                features = dinov2_vitb14(images)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # val
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                features = dinov2_vitb14(images)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # tracking best model
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
    
    # loading best model state
    model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'history': history,
        'epochs_trained': epoch + 1
    }


##B GRID SEARCH ------------------

print("=" * 80)
print("Running Grid Search (Warm Start)")
print("=" * 80)
print()

param_names = list(HYPERPARAM_GRID.keys())
param_values = [HYPERPARAM_GRID[name] for name in param_names]
all_combinations = list(itertools.product(*param_values))

results = []
best_overall_acc = 0.0
best_hyperparams = None
best_model = None

for i, combination in enumerate(all_combinations):
    hyperparams = dict(zip(param_names, combination))
    
    print(f"\n[{i+1}/{total_combinations}] Testing hyperparameters:")
    for param, value in hyperparams.items():
        print(f"  {param}: {value}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # training model (with warm start)
    result = train_model(hyperparams, train_loader, val_loader, pretrained_state_dict)
    
    print(f"→ Val Accuracy: {result['best_val_acc']:.4f} ({result['best_val_acc']*100:.2f}%)")
    print(f"→ Val Loss: {result['best_val_loss']:.4f}")
    print(f"→ Epochs: {result['epochs_trained']}")
    
    # storing results
    results.append({
        **hyperparams,
        'val_accuracy': result['best_val_acc'],
        'val_loss': result['best_val_loss'],
        'epochs_trained': result['epochs_trained']
    })
    
    # tracking best
    if result['best_val_acc'] > best_overall_acc:
        best_overall_acc = result['best_val_acc']
        best_hyperparams = hyperparams
        best_model = result['model']
        print(f"  ✨ NEW BEST MODEL! Val Acc: {best_overall_acc:.4f}")

print()
print("=" * 80)
print("Grid Search Complete!")
print("=" * 80)
print()

# saving all results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_accuracy', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'gridsearch_results.csv'), index=False)

print("Top 10 Hyperparameter Combinations:")
print(results_df.head(10).to_string(index=False))
print()

print("Best Hyperparameters:")
for param, value in best_hyperparams.items():
    print(f"  {param}: {value}")
print(f"  Validation Accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
print()

# saving best hyperparameters
with open(os.path.join(OUTPUT_DIR, 'best_hyperparameters.json'), 'w') as f:
    json.dump({
        **best_hyperparams,
        'val_accuracy': float(best_overall_acc)
    }, f, indent=2)

# saving best model
torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
print(f"✅ Best model saved")
print()

# visualising grid search results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, param in enumerate(param_names):
    if idx < len(axes):
        grouped = results_df.groupby(param)['val_accuracy'].mean()
        axes[idx].bar(range(len(grouped)), grouped.values, color='steelblue', alpha=0.7)
        axes[idx].set_xticks(range(len(grouped)))
        axes[idx].set_xticklabels(grouped.index, rotation=45, ha='right')
        axes[idx].set_ylabel('Mean Val Accuracy')
        axes[idx].set_title(f'Effect of {param}')
        axes[idx].grid(alpha=0.3, axis='y')

for idx in range(len(param_names), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gridsearch_visualization.png'), dpi=300, bbox_inches='tight')
print("✅ Saved gridsearch_visualization.png")
print()

## EVAL MODEL ON TEST ------------------

print("=" * 80)
print("Evaluating Best Model on Test Set")
print("=" * 80)
print()

best_model.eval()

test_loader = DataLoader(
    test_dataset, 
    batch_size=best_hyperparams['batch_size'], 
    shuffle=False, 
    num_workers=4
)

all_labels = []
all_preds = []
all_probs = []
all_features = []
all_image_paths = []

with torch.no_grad():
    for images, labels, image_paths in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        features = dinov2_vitb14(images)
        outputs = best_model(features)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_features.extend(features.cpu().numpy())
        all_image_paths.extend(image_paths)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_features = np.array(all_features)

## COMPUTING METRICS ------------------

print()
print("=" * 80)
print("Test Set Metrics")
print("=" * 80)
print()

accuracy = accuracy_score(all_labels, all_preds)
print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(len(TARGET_CLASSES)))
print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
print()

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

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro', zero_division=0
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted', zero_division=0
)

print("Aggregate Metrics:")
print(f"Macro Avg - Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}")
print(f"Weighted Avg - Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f}, F1: {weighted_f1:.4f}")
print()

cm = confusion_matrix(all_labels, all_preds, labels=range(len(TARGET_CLASSES)))
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES))
print()

# saving
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'test_per_class_metrics.csv'), index=False)
summary_df = pd.DataFrame({
    'Metric': ['Top-1 Accuracy', 'Top-3 Accuracy', 
               'Macro Precision', 'Macro Recall', 'Macro F1',
               'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
    'Score': [accuracy, top3_acc, macro_p, macro_r, macro_f1,
              weighted_p, weighted_r, weighted_f1]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'test_summary_metrics.csv'), index=False)

predictions_df = pd.DataFrame({
    'image_path': all_image_paths,
    'true_label': [IDX_TO_CLASS[i] for i in all_labels],
    'pred_label': [IDX_TO_CLASS[i] for i in all_preds],
    'confidence': np.max(all_probs, axis=1),
    **{f'prob_{cls}': all_probs[:, idx] for idx, cls in enumerate(TARGET_CLASSES)}
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

print(classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0))

print()
print("=" * 80)
print("WARM-START GRID SEARCH COMPLETE")
print("=" * 80)
print()
print(f"📊 FINAL RESULTS:")
print(f"\n🔥 Warm-started from: {PRETRAINED_MODEL}")
print(f"\n🎯 Best Hyperparameters:")
for param, value in best_hyperparams.items():
    print(f"  {param}: {value}")
print(f"\n📈 Validation: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
print(f"🧪 Test Top-1:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🧪 Test Top-3:  {top3_acc:.4f} ({top3_acc*100:.2f}%)")
print(f"📁 Results: {OUTPUT_DIR}")
print("=" * 80)
