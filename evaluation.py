import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import seaborn as sns
from tqdm import tqdm
import json
import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import sys
sys.path.append(r'D:\A15-CNN')

from training import A15CNN, ChestXRayDataset

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder untuk handle numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def load_model_for_evaluation():
    """Load model untuk evaluation"""
    possible_paths = [
        r"D:\A15-CNN\models\final_model.pth",
    ]
    
    MODEL_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    if MODEL_PATH is None:
        print("❌ Tidak ada file model yang ditemukan!")
        print("Cek path berikut:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    model = A15CNN(num_classes=2).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"✅ Model loaded dari {MODEL_PATH}")
            print(f"Validation accuracy saat training: {val_acc}%")
        else:
            # Jika file langsung state_dict
            model.load_state_dict(checkpoint)
            print(f"✅ Model loaded dari {MODEL_PATH}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.eval()
    return model

def rigorous_evaluation():
    """Evaluasi rigorous pada test set"""
    
    TEST_DIR = r"D:\A15-CNN\data\test"
    
    # Load model
    model = load_model_for_evaluation()
    if model is None:
        print("Gagal load model. Pastikan file model ada.")
        return None
    
    # Test dataset
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory tidak ditemukan: {TEST_DIR}")
        return None
        
    test_dataset = ChestXRayDataset(TEST_DIR, transform=test_transform)
    
    if len(test_dataset) == 0:
        print("❌ Tidak ada data test yang ditemukan!")
        return None
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"Test samples: {len(test_dataset)}")
    print("Memulai rigorous evaluation...")
    
    # Predict satu per satu untuk memastikan akurasi
    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        try:
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            with torch.no_grad():
                output = model(image)
                probability = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
            all_predictions.append(predicted.cpu().item())
            all_labels.append(label)
            all_probabilities.append(probability.cpu().numpy()[0])
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if len(all_predictions) == 0:
        print("❌ Tidak ada prediksi yang berhasil!")
        return None
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate comprehensive metrics
    accuracy = float(np.mean(all_predictions == all_labels))
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)
    
    # Cohen's Kappa
    kappa = float(cohen_kappa_score(all_labels, all_predictions))
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
    roc_auc = float(auc(fpr, tpr))
    
    # Per-class metrics
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=['Normal', 'Tuberkulosis'],
        digits=4
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print(f"\n" + "="*60)
    print("RIGOROUS EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(class_report)
    
    # Per-class accuracy
    normal_correct = int(np.sum((all_labels == 0) & (all_predictions == 0)))
    normal_total = int(np.sum(all_labels == 0))
    tb_correct = int(np.sum((all_labels == 1) & (all_predictions == 1)))
    tb_total = int(np.sum(all_labels == 1))
    
    print(f"\nPer-class Accuracy:")
    print(f"Normal:       {normal_correct}/{normal_total} = {100.*normal_correct/normal_total:.2f}%")
    print(f"Tuberkulosis: {tb_correct}/{tb_total} = {100.*tb_correct/tb_total:.2f}%")
    
    # Plot basic results
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_probability_distribution(all_labels, all_probabilities)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
        'confusion_matrix': cm.tolist(),
        'test_samples': len(test_dataset),
        'normal_samples': normal_total,
        'tb_samples': tb_total,
        'normal_correct': normal_correct,
        'tb_correct': tb_correct
    }
    
    return results

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Tuberkulosis'],
                yticklabels=['Normal', 'Tuberkulosis'])
    plt.title('Confusion Matrix - A15-CNN Tuberculosis Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(r'D:\A15-CNN\assets\confusion_matrix_rigorous.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC Curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - A15-CNN Tuberculosis Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'D:\A15-CNN\assets\roc_curve_rigorous.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution(labels, probabilities):
    """Plot probability distribution for each class"""
    plt.figure(figsize=(12, 5))
    
    # Normal class probabilities
    plt.subplot(1, 2, 1)
    normal_probs = probabilities[labels == 0, 1]
    plt.hist(normal_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Probability Distribution - Normal Cases')
    plt.xlabel('Probability of Tuberculosis')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # TB class probabilities
    plt.subplot(1, 2, 2)
    tb_probs = probabilities[labels == 1, 1]
    plt.hist(tb_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.title('Probability Distribution - Tuberculosis Cases')
    plt.xlabel('Probability of Tuberculosis')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'D:\A15-CNN\assets\probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_analysis(results):
    """Plot analisis detail hasil evaluasi"""
    
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    probabilities = np.array(results['probabilities'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence Distribution for Misclassifications
    misclassified = np.where(predictions != labels)[0]
    correct = np.where(predictions == labels)[0]
    
    if len(misclassified) > 0:
        misclass_confidences = [max(prob) for prob in probabilities[misclassified]]
        correct_confidences = [max(prob) for prob in probabilities[correct]]
        
        axes[0,0].hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0,0].hist(misclass_confidences, bins=30, alpha=0.7, label='Misclassified', color='red')
        axes[0,0].set_xlabel('Prediction Confidence')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Confidence Distribution: Correct vs Misclassified')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    else:
        axes[0,0].text(0.5, 0.5, 'No Misclassifications\nPerfect Model!', 
                      ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12)
        axes[0,0].set_title('Confidence Distribution')
    
    # 2. Error Analysis by Class
    error_types = {
        'False Positive': int(np.sum((labels == 0) & (predictions == 1))),
        'False Negative': int(np.sum((labels == 1) & (predictions == 0)))
    }
    
    axes[0,1].bar(error_types.keys(), error_types.values(), color=['red', 'orange'])
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Error Type Analysis')
    for i, v in enumerate(error_types.values()):
        axes[0,1].text(i, v + 0.1, str(v), ha='center', va='bottom')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Probability Calibration
    prob_true, prob_pred = calibration_curve(labels, probabilities[:, 1], n_bins=10)
    
    axes[1,0].plot(prob_pred, prob_true, 's-', label='A15-CNN', markersize=6)
    axes[1,0].plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    axes[1,0].set_xlabel('Mean Predicted Probability')
    axes[1,0].set_ylabel('Fraction of Positives')
    axes[1,0].set_title('Probability Calibration Curve')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Performance Summary
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        results['accuracy'],
        results['precision'], 
        results['recall'],
        results['f1']
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = axes[1,1].bar(metrics, values, color=colors, alpha=0.8)
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Performance Metrics Summary')
    axes[1,1].set_ylim(0.95, 1.0)
    
    # Add value labels on bars
    for bar, v in zip(bars, values):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.002,
                      f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'D:\A15-CNN\assets\detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_performance(results):
    """Plot comprehensive performance visualization"""
    
    plt.figure(figsize=(14, 10))
    
    # 1. Main performance metrics radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'], 
        results['f1'],
        results['roc_auc']
    ]
    
    # Normalize for radar chart (all already 0-1)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]  # Complete the circle
    
    ax1 = plt.subplot(2, 2, 1, polar=True)
    ax1.plot(angles, values, 'o-', linewidth=2, label='A15-CNN', color='#FF6B6B')
    ax1.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax1.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Performance Radar Chart', size=14, fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # 2. Confidence distribution for both classes
    ax2 = plt.subplot(2, 2, 2)
    labels_np = np.array(results['labels'])
    probabilities_np = np.array(results['probabilities'])
    
    normal_confidences = probabilities_np[labels_np == 0, 1]
    tb_confidences = probabilities_np[labels_np == 1, 1]
    
    ax2.hist(normal_confidences, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    ax2.hist(tb_confidences, bins=30, alpha=0.7, label='Tuberculosis', color='red', density=True)
    ax2.set_xlabel('Predicted Probability of Tuberculosis')
    ax2.set_ylabel('Density')
    ax2.set_title('Confidence Distribution by True Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    ax3 = plt.subplot(2, 2, 3)
    precision_curve, recall_curve, _ = roc_curve(labels_np, probabilities_np[:, 1])
    ax3.plot(recall_curve, precision_curve, color='purple', lw=2)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 4. Metrics comparison table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    metric_data = [
        ['Accuracy', f"{results['accuracy']:.4f}"],
        ['Precision', f"{results['precision']:.4f}"],
        ['Recall', f"{results['recall']:.4f}"],
        ['F1-Score', f"{results['f1']:.4f}"],
        ['AUC-ROC', f"{results['roc_auc']:.4f}"],
        ["Cohen's Kappa", f"{results['kappa']:.4f}"]
    ]
    
    table = ax4.table(cellText=metric_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax4.set_title('Performance Metrics Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(r'D:\A15-CNN\assets\comprehensive_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_latex_tables(results):
    """Generate LaTeX tables untuk paper"""
    
    cm = results['confusion_matrix']
    
    # Performance Table
    performance_latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Performance Metrics of A15-CNN Model for Tuberculosis Detection}}
\\label{{tab:performance}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Accuracy & {results['accuracy']:.4f} ({results['accuracy']*100:.2f}\\%%) \\\\
Precision & {results['precision']:.4f} \\\\
Recall (Sensitivity) & {results['recall']:.4f} \\\\
F1-Score & {results['f1']:.4f} \\\\
Cohen's Kappa & {results['kappa']:.4f} \\\\
ROC AUC & {results['roc_auc']:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    # Confusion Matrix Table
    cm_latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Confusion Matrix of A15-CNN Model}}
\\label{{tab:confusion}}
\\begin{{tabular}}{{lcc}}
\\hline
& \\textbf{{Predicted Normal}} & \\textbf{{Predicted Tuberculosis}} \\\\
\\hline
\\textbf{{Actual Normal}} & {cm[0][0]} & {cm[0][1]} \\\\
\\textbf{{Actual Tuberculosis}} & {cm[1][0]} & {cm[1][1]} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    # Detailed Metrics Table
    detailed_latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Detailed Classification Report}}
\\label{{tab:detailed}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Class}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} & \\textbf{{Support}} \\\\
\\hline
Normal & {results['precision']:.4f} & {1 - results['recall']:.4f} & {results['f1']:.4f} & {results['normal_samples']} \\\\
Tuberculosis & {results['recall']:.4f} & {results['precision']:.4f} & {results['f1']:.4f} & {results['tb_samples']} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    latex_content = f"""
% A15-CNN Tuberculosis Detection Results
% Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== PERFORMANCE TABLE ===
{performance_latex}

=== CONFUSION MATRIX ===
{cm_latex}

=== DETAILED METRICS ===
{detailed_latex}
"""
    
    with open(r'D:\A15-CNN\assets\latex_tables.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print("✅ LaTeX tables generated: latex_tables.tex")

def save_results_json(results):
    """Save results to JSON file untuk documentation"""
    
    results_export = {
        'model_name': 'A15-CNN',
        'timestamp': datetime.datetime.now().isoformat(),
        'dataset': {
            'test_samples': results['test_samples'],
            'normal_samples': results['normal_samples'],
            'tuberculosis_samples': results['tb_samples']
        },
        'performance_metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1'],
            'roc_auc': results['roc_auc'],
            'cohens_kappa': results['kappa']
        },
        'classification_results': {
            'normal_correct': results['normal_correct'],
            'normal_total': results['normal_samples'],
            'tb_correct': results['tb_correct'],
            'tb_total': results['tb_samples'],
            'normal_accuracy': results['normal_correct'] / results['normal_samples'],
            'tb_accuracy': results['tb_correct'] / results['tb_samples']
        },
        'confusion_matrix': results['confusion_matrix']
    }
    
    try:
        with open(r'D:\A15-CNN\assets\evaluation_results.json', 'w') as f:
            json.dump(results_export, f, indent=2, cls=NumpyEncoder)
        print("✅ Results saved to JSON: evaluation_results.json")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

def analyze_misclassifications(results):
    """Analisis gambar yang salah diklasifikasi"""
    
    TEST_DIR = r"D:\A15-CNN\data\test"
    
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    probabilities = np.array(results['probabilities'])
    
    # Find misclassified indices
    misclassified = np.where(predictions != labels)[0]
    
    if len(misclassified) == 0:
        print("\n🎉 Tidak ada misclassification ditemukan! Model sempurna!")
        return
    
    print(f"\n=== MISCLASSIFICATION ANALYSIS ===")
    print(f"Total misclassifications: {len(misclassified)}")
    print(f"Error rate: {100.*len(misclassified)/len(predictions):.2f}%")
    
    # Sample some misclassifications
    sample_size = min(5, len(misclassified))
    print(f"\nSample {sample_size} misclassifications:")
    
    for i in range(sample_size):
        idx = misclassified[i]
        true_label = labels[idx]
        pred_label = predictions[idx]
        prob = probabilities[idx]
        
        true_class = "Normal" if true_label == 0 else "Tuberkulosis"
        pred_class = "Normal" if pred_label == 0 else "Tuberkulosis"
        
        print(f"\nMisclassification {i+1}:")
        print(f"  True: {true_class}, Predicted: {pred_class}")
        print(f"  Probabilities: Normal={prob[0]:.4f}, TB={prob[1]:.4f}")
        print(f"  Confidence: {max(prob):.4f}")

def cross_validation_evaluation():
    """Simple cross-validation style evaluation"""
    
    print("\n=== CROSS-VALIDATION STYLE EVALUATION ===")
    
    # Load model
    model = load_model_for_evaluation()
    if model is None:
        print("❌ Tidak dapat melakukan cross-validation: model tidak tersedia")
        return
    
    # Evaluate on all splits
    splits = ['train', 'val', 'test']
    base_dir = r"D:\A15-CNN\data"
    
    print("\nSplit-wise Performance:")
    print("-" * 40)
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ Directory tidak ditemukan: {split_dir}")
            continue
            
        dataset = ChestXRayDataset(split_dir, transform=test_transform)
        if len(dataset) == 0:
            print(f"❌ Tidak ada data di {split_dir}")
            continue
            
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"{split.upper():<10} : {accuracy:6.2f}% ({correct:4d}/{total:4d})")

def main():
    """Main function dengan error handling"""
    print("=== A15-CNN: COMPREHENSIVE EVALUATION ===")
    print("=" * 50)
    
    try:
        # 1. Rigorous evaluation on test set
        print("\n1. Running rigorous evaluation...")
        results = rigorous_evaluation()
        
        if results is not None:
            # 2. Detailed analysis and visualizations
            print("\n2. Generating detailed analysis...")
            plot_detailed_analysis(results)
            plot_comprehensive_performance(results)
            
            # 3. Research paper materials
            print("\n3. Generating research paper materials...")
            generate_latex_tables(results)
            save_results_json(results)
            
            # 4. Analyze misclassifications
            print("\n4. Analyzing misclassifications...")
            analyze_misclassifications(results)
        else:
            print("❌ Skipping further analysis: no results available")
        
        # 5. Cross-validation style evaluation
        print("\n5. Running cross-validation evaluation...")
        cross_validation_evaluation()
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print("="*60)
    print("Generated files:")
    generated_files = [
        "confusion_matrix_rigorous.png",
        "roc_curve_rigorous.png", 
        "probability_distribution.png",
        "detailed_analysis.png",
        "comprehensive_performance.png",
        "latex_tables.tex",
        "evaluation_results.json"
    ]
    
    for file in generated_files:
        if os.path.exists(os.path.join(r'D:\A15-CNN\assets', file)):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (not generated)")

if __name__ == "__main__":
    main()