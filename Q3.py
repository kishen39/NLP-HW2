import numpy as np

def evaluate_confusion_matrix(confusion_matrix, class_names):
    """
    Evaluate a multi-class confusion matrix and compute various metrics
    
    Args:
        confusion_matrix: 2D numpy array or list of lists representing the confusion matrix
        class_names: List of class names in order
    
    Returns:
        Dictionary containing all computed metrics
    """
    
    # Convert to numpy array for easier computation
    cm = np.array(confusion_matrix)
    n_classes = len(class_names)
    
    # Initialize results dictionary
    results = {
        'per_class': {},
        'macro': {},
        'micro': {}
    }
    
    # Per-class precision and recall
    for i, class_name in enumerate(class_names):
        # True positives for this class
        tp = cm[i, i]
        
        # False positives for this class (sum of column i excluding diagonal)
        fp = np.sum(cm[:, i]) - tp
        
        # False negatives for this class (sum of row i excluding diagonal)
        fn = np.sum(cm[i, :]) - tp
        
        # True negatives for this class
        tn = np.sum(cm) - tp - fp - fn
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results['per_class'][class_name] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # Macro-averaged precision and recall
    macro_precision = np.mean([results['per_class'][cls]['precision'] for cls in class_names])
    macro_recall = np.mean([results['per_class'][cls]['recall'] for cls in class_names])
    
    results['macro']['precision'] = macro_precision
    results['macro']['recall'] = macro_recall
    
    # Micro-averaged precision and recall
    total_tp = sum(results['per_class'][cls]['tp'] for cls in class_names)
    total_fp = sum(results['per_class'][cls]['fp'] for cls in class_names)
    total_fn = sum(results['per_class'][cls]['fn'] for cls in class_names)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    results['micro']['precision'] = micro_precision
    results['micro']['recall'] = micro_recall
    
    # Additional overall metrics
    results['confusion_matrix'] = cm
    results['total_samples'] = np.sum(cm)
    results['accuracy'] = np.trace(cm) / np.sum(cm)
    
    return results

def print_results(results, class_names):
    """Print the evaluation results in a clear format"""
    
    print("=" * 60)
    print("CONFUSION MATRIX EVALUATION RESULTS")
    print("=" * 60)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"{'System \\ Gold':<12}", end="")
    for name in class_names:
        print(f"{name:<8}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:<12}", end="")
        for j in range(len(class_names)):
            print(f"{results['confusion_matrix'][i, j]:<8}", end="")
        print()
    
    print(f"\nTotal samples: {results['total_samples']}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    # Print per-class metrics
    print("\n" + "=" * 40)
    print("PER-CLASS METRICS")
    print("=" * 40)
    
    for class_name in class_names:
        metrics = results['per_class'][class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f} "
              f"(TP: {metrics['tp']}, FP: {metrics['fp']})")
        print(f"  Recall:    {metrics['recall']:.4f} "
              f"(TP: {metrics['tp']}, FN: {metrics['fn']})")
    
    # Print macro-averaged metrics
    print("\n" + "=" * 40)
    print("MACRO-AVERAGED METRICS")
    print("=" * 40)
    print(f"Precision: {results['macro']['precision']:.4f}")
    print(f"Recall:    {results['macro']['recall']:.4f}")
    
    # Print micro-averaged metrics
    print("\n" + "=" * 40)
    print("MICRO-AVERAGED METRICS")
    print("=" * 40)
    print(f"Precision: {results['micro']['precision']:.4f}")
    print(f"Recall:    {results['micro']['recall']:.4f}")
    
    print("\n" + "=" * 60)

# Main execution
if __name__ == "__main__":
    # Define the confusion matrix and class names
    confusion_matrix = [
        [5, 10, 5],   # Cat predictions
        [15, 20, 10], # Dog predictions  
        [0, 15, 10]   # Rabbit predictions
    ]
    
    class_names = ['Cat', 'Dog', 'Rabbit']
    
    # Evaluate the confusion matrix
    results = evaluate_confusion_matrix(confusion_matrix, class_names)
    
    # Print all results
    print_results(results, class_names)