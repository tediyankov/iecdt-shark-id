
## CLIP-Based Shark Species Classification
## job: 66974960

## libraries 
import os
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# CONFIG ------------------

LABELS_CSV = "./data/labels.csv"
BASE_DIR = "."

# output dirs
OUTPUT_DIR_ZERO = "./results/clip_evaluation_zero_shot"
OUTPUT_DIR_FEW = "./results/clip_evaluation_few_shot"

os.makedirs(OUTPUT_DIR_ZERO, exist_ok=True)
os.makedirs(OUTPUT_DIR_FEW, exist_ok=True)

# target classes (all 4 species)
TARGET_CLASSES = ['grey_reef_shark', 'blacktip_reef_shark', 'whitetip_reef_shark', 'tawny_nurse_shark']

# few-shot param
FEW_SHOT_K = 10 

# TEXT PROMPT TEMPLATE ------------------

# multiple prompt templates for ensemble
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a photo of a {}, a type of shark",
    "a underwater photo of a {}",
    "{} swimming in the ocean",
    "a {} in its natural habitat",
]

# species name variations for better matching
SPECIES_VARIATIONS = {
    'grey_reef_shark': [
        'grey reef shark',
        'gray reef shark', 
        'Carcharhinus amblyrhynchos',
    ],
    'blacktip_reef_shark': [
        'blacktip reef shark',
        'black tip reef shark',
        'Carcharhinus melanopterus',
    ],
    'whitetip_reef_shark': [
        'whitetip reef shark',
        'white tip reef shark',
        'Triaenodon obesus',
    ],
    'tawny_nurse_shark': [
        'tawny nurse shark',
        'nurse shark',
        'Nebrius ferrugineus',
    ]
}

## LOADING CLIP MODEL ------------------

print("=" * 80)
print("CLIP-Based Shark Species Classification")
print("=" * 80)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("Loading CLIP model (ViT-L/14)...")
model, preprocess = clip.load("ViT-L/14", device=device)
print("✅ Model loaded")
print()

## LOADING AND PREPPING DATA ------------------

print("=" * 80)
print("Loading Ground Truth Labels")
print("=" * 80)

labels_df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(labels_df)} total labeled images")
print(f"Full class distribution:\n{labels_df['species'].value_counts()}\n")

# filtering to target classes
labels_df = labels_df[labels_df['species'].isin(TARGET_CLASSES)].copy()
print(f"Using {len(labels_df)} images for evaluation (4 species)")
print(f"Class distribution:\n{labels_df['species'].value_counts()}")
print()

#PREPPING TEXT EMBEDDINGS ------------------

print("=" * 80)
print("ZERO-SHOT: Preparing Text Embeddings")
print("=" * 80)

# creating all text prompts
all_prompts = []
prompt_to_class = {}

for cls in TARGET_CLASSES:
    for variation in SPECIES_VARIATIONS[cls]:
        for template in PROMPT_TEMPLATES:
            prompt = template.format(variation)
            all_prompts.append(prompt)
            prompt_to_class[prompt] = cls

print(f"Created {len(all_prompts)} text prompts ({len(all_prompts)//len(TARGET_CLASSES)} per class)")
print("\nExample prompts:")
for cls in TARGET_CLASSES[:2]:
    print(f"  {cls}:")
    prompts = [p for p in all_prompts if prompt_to_class[p] == cls][:3]
    for p in prompts:
        print(f"    - '{p}'")
print()

# encoding text prompts
print("Encoding text prompts...")
text_tokens = clip.tokenize(all_prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# avg embeddings per class for ensemble
class_text_features = {}
for cls in TARGET_CLASSES:
    cls_prompts = [i for i, p in enumerate(all_prompts) if prompt_to_class[p] == cls]
    cls_features = text_features[cls_prompts].mean(dim=0)
    cls_features /= cls_features.norm()
    class_text_features[cls] = cls_features

print("✅ Text embeddings prepared")
print()

## ZERO SHOT CLASSIFICATION ------------------

print("=" * 80)
print("ZERO-SHOT: Classifying Images")
print("=" * 80)

zero_shot_results = []
failed = 0

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Zero-shot classification"):
    img_path = os.path.join(BASE_DIR, row['image_path'])
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found")
        failed += 1
        continue
    
    try:
        # loading and preprocess image
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        # encoding image
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # computing similarity to each class
        similarities = {}
        for cls in TARGET_CLASSES:
            sim = (image_features @ class_text_features[cls].unsqueeze(1)).item()
            similarities[cls] = sim
        
        # getting preds
        pred_cls = max(similarities, key=similarities.get)
        confidence = similarities[pred_cls]
        
        zero_shot_results.append({
            'image_path': row['image_path'],
            'true_species': row['species'],
            'pred_species': pred_cls,
            'confidence': confidence,
            **{f'sim_{cls}': similarities[cls] for cls in TARGET_CLASSES}
        })
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        failed += 1
        continue

print(f"\n✅ Zero-shot classification complete!")
print(f"Successfully classified: {len(zero_shot_results)}")
print(f"Failed: {failed}")
print()

zero_shot_df = pd.DataFrame(zero_shot_results)
zero_shot_df.to_csv(os.path.join(OUTPUT_DIR_ZERO, 'predictions.csv'), index=False)

## EVAL OF ZERO SHOT ------------------

def evaluate_and_save(results_df, output_dir, method_name):
    """Compute metrics and create visualizations"""
    
    print("=" * 80)
    print(f"{method_name}: Computing Metrics")
    print("=" * 80)
    
    y_true = results_df['true_species'].values
    y_pred = results_df['pred_species'].values
    confidences = results_df['confidence'].values
    
    # overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, zero_division=0
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
    
    # aggregate metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, average='weighted', zero_division=0
    )
    
    print("Aggregate Metrics:")
    print(f"Macro Avg - Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}")
    print(f"Weighted Avg - Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f}, F1: {weighted_f1:.4f}")
    print()
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES))
    print()
    
    # saving metrics
    metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',
                   'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Score': [accuracy, macro_p, macro_r, macro_f1, weighted_p, weighted_r, weighted_f1]
    })
    summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    # classification report
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, labels=TARGET_CLASSES, zero_division=0))
    print()
    
    # conf analysis
    print("Confidence Statistics:")
    print(f"Mean: {confidences.mean():.4f}")
    print(f"Median: {np.median(confidences):.4f}")
    print(f"Std: {confidences.std():.4f}")
    print()
    
    results_df['correct'] = results_df['true_species'] == results_df['pred_species']
    correct_conf = results_df[results_df['correct']]['confidence']
    incorrect_conf = results_df[~results_df['correct']]['confidence']
    
    if len(correct_conf) > 0:
        print(f"Correct predictions ({len(correct_conf):3d}): Mean confidence = {correct_conf.mean():.4f}")
    if len(incorrect_conf) > 0:
        print(f"Incorrect predictions ({len(incorrect_conf):3d}): Mean confidence = {incorrect_conf.mean():.4f}")
    print()
    
    # error analysis
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        print(f"Errors: {len(errors)} / {len(results_df)} ({len(errors)/len(results_df)*100:.1f}%)")
        print("Error breakdown:")
        error_summary = errors.groupby(['true_species', 'pred_species']).size().reset_index(name='count')
        print(error_summary.to_string(index=False))
        print()
        errors.to_csv(os.path.join(output_dir, 'errors.csv'), index=False)
    
    # viz
    print("Creating visualizations...")
    
    # conf matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.replace('_', ' ').title() for c in TARGET_CLASSES],
                yticklabels=[c.replace('_', ' ').title() for c in TARGET_CLASSES],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {method_name}\n(n={len(results_df)})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # per-class metrics
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
    ax.set_title(f'Per-Class Performance - {method_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in TARGET_CLASSES], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    
    print(f"✅ Results saved to {output_dir}\n")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

# eval 0 shot
zero_metrics = evaluate_and_save(zero_shot_df, OUTPUT_DIR_ZERO, "CLIP Zero-Shot")

## FEW SHOT CLASSIFICATION ------------------

print("=" * 80)
print(f"FEW-SHOT: Selecting {FEW_SHOT_K} Support Examples per Class")
print("=" * 80)

# splitting data: few-shot support set + test set
support_examples = {}
test_indices = []

for cls in TARGET_CLASSES:
    cls_data = labels_df[labels_df['species'] == cls]
    
    if len(cls_data) < FEW_SHOT_K:
        print(f"Warning: {cls} has only {len(cls_data)} samples, using all as support")
        n_support = len(cls_data)
    else:
        n_support = FEW_SHOT_K
    
    # random sample support examples
    support_idx = cls_data.sample(n=n_support, random_state=42).index.tolist()
    support_examples[cls] = support_idx
    
    # rest goes to test set
    test_idx = cls_data.drop(support_idx).index.tolist()
    test_indices.extend(test_idx)
    
    print(f"  {cls:25s}: {n_support} support, {len(test_idx)} test")

print(f"\nTotal: {sum(len(v) for v in support_examples.values())} support, {len(test_indices)} test")
print()

# compute support prototypes (avg image embeddings per class)
print("Computing few-shot prototypes...")
class_prototypes = {}

for cls in TARGET_CLASSES:
    cls_embeddings = []
    
    for idx in tqdm(support_examples[cls], desc=f"  {cls}", leave=False):
        row = labels_df.loc[idx]
        img_path = os.path.join(BASE_DIR, row['image_path'])
        
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            cls_embeddings.append(image_features.cpu())
        except Exception as e:
            print(f"Error loading support image {img_path}: {e}")
            continue
    
    if len(cls_embeddings) > 0:
        prototype = torch.stack(cls_embeddings).mean(dim=0).to(device)
        prototype /= prototype.norm()
        class_prototypes[cls] = prototype
        print(f"  ✅ {cls}: prototype from {len(cls_embeddings)} images")
    else:
        print(f"  ⚠️  {cls}: no valid support images!")

print()

# few-shot classification on test set
print("=" * 80)
print("FEW-SHOT: Classifying Test Images")
print("=" * 80)

few_shot_results = []
failed_few = 0

test_df = labels_df.loc[test_indices]

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Few-shot classification"):
    img_path = os.path.join(BASE_DIR, row['image_path'])
    
    if not os.path.exists(img_path):
        failed_few += 1
        continue
    
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # computing similarity to class prototypes
        similarities = {}
        for cls in TARGET_CLASSES:
            if cls in class_prototypes:
                sim = (image_features @ class_prototypes[cls].T).item()
                similarities[cls] = sim
            else:
                # penalising missing prototypes
                similarities[cls] = -1.0
        
        pred_cls = max(similarities, key=similarities.get)
        confidence = similarities[pred_cls]
        
        few_shot_results.append({
            'image_path': row['image_path'],
            'true_species': row['species'],
            'pred_species': pred_cls,
            'confidence': confidence,
            **{f'sim_{cls}': similarities[cls] for cls in TARGET_CLASSES}
        })
        
    except Exception as e:
        print(f"Error: {e}")
        failed_few += 1
        continue

print(f"\n✅ Few-shot classification complete!")
print(f"Successfully classified: {len(few_shot_results)}")
print(f"Failed: {failed_few}")
print()

few_shot_df = pd.DataFrame(few_shot_results)
few_shot_df.to_csv(os.path.join(OUTPUT_DIR_FEW, 'predictions.csv'), index=False)

# eval few-shot
few_metrics = evaluate_and_save(few_shot_df, OUTPUT_DIR_FEW, "CLIP Few-Shot")

## FINAL COMPARISON ------------------

print("=" * 80)
print("FINAL COMPARISON: Zero-Shot vs Few-Shot")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Method': ['CLIP Zero-Shot', 'CLIP Few-Shot'],
    'Test Samples': [len(zero_shot_df), len(few_shot_df)],
    'Accuracy': [zero_metrics['accuracy'], few_metrics['accuracy']],
    'Macro F1': [zero_metrics['macro_f1'], few_metrics['macro_f1']],
    'Weighted F1': [zero_metrics['weighted_f1'], few_metrics['weighted_f1']]
})

print(comparison_df.to_string(index=False))
print()

comparison_df.to_csv(os.path.join(BASE_DIR, 'clip_comparison.csv'), index=False)
print(f"✅ Comparison saved to clip_comparison.csv")
print()
print("=" * 80)
print("all evals complete yay!")
print("=" * 80)
