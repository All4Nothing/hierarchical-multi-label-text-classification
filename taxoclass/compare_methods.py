"""
Compare two inference methods on a small sample
"""
import numpy as np
import csv
from collections import Counter


def load_submission(filepath):
    """Load submission CSV file"""
    predictions = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['id']
            labels = [int(x) for x in row['labels'].split(',')]
            predictions[pid] = labels
    return predictions


def compare_submissions(file1, file2, sample_ids=None):
    """
    Compare two submission files
    
    Args:
        file1: Path to first submission
        file2: Path to second submission
        sample_ids: List of sample IDs to compare (None = all)
    """
    print("="*80)
    print("SUBMISSION COMPARISON")
    print("="*80)
    
    pred1 = load_submission(file1)
    pred2 = load_submission(file2)
    
    print(f"\nFile 1: {file1}")
    print(f"  Total samples: {len(pred1)}")
    
    print(f"\nFile 2: {file2}")
    print(f"  Total samples: {len(pred2)}")
    
    # Get common samples
    common_ids = set(pred1.keys()) & set(pred2.keys())
    print(f"\nCommon samples: {len(common_ids)}")
    
    if sample_ids:
        sample_ids = [str(sid) for sid in sample_ids]
        common_ids = [sid for sid in sample_ids if sid in common_ids]
    else:
        # Take first 10 for comparison
        common_ids = sorted(list(common_ids), key=int)[:10]
    
    # Compare predictions
    print("\n" + "="*80)
    print("SAMPLE-BY-SAMPLE COMPARISON")
    print("="*80)
    
    differences = 0
    for sample_id in common_ids:
        labels1 = set(pred1[sample_id])
        labels2 = set(pred2[sample_id])
        
        if labels1 != labels2:
            differences += 1
            print(f"\nSample {sample_id}:")
            print(f"  Method 1: {sorted(labels1)}")
            print(f"  Method 2: {sorted(labels2)}")
            
            # Overlap analysis
            common = labels1 & labels2
            only1 = labels1 - labels2
            only2 = labels2 - labels1
            
            if common:
                print(f"  Common:   {sorted(common)}")
            if only1:
                print(f"  Only in Method 1: {sorted(only1)}")
            if only2:
                print(f"  Only in Method 2: {sorted(only2)}")
    
    if differences == 0:
        print("\n✅ All compared samples have identical predictions!")
    else:
        print(f"\n⚠️  {differences}/{len(common_ids)} samples differ ({100*differences/len(common_ids):.1f}%)")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    # Label count distribution
    label_counts1 = [len(pred1[sid]) for sid in common_ids]
    label_counts2 = [len(pred2[sid]) for sid in common_ids]
    
    print("\nLabels per sample:")
    print(f"  Method 1: min={min(label_counts1)}, max={max(label_counts1)}, avg={np.mean(label_counts1):.2f}")
    print(f"  Method 2: min={min(label_counts2)}, max={max(label_counts2)}, avg={np.mean(label_counts2):.2f}")
    
    # Unique classes
    all_classes1 = set()
    all_classes2 = set()
    for sid in common_ids:
        all_classes1.update(pred1[sid])
        all_classes2.update(pred2[sid])
    
    print(f"\nUnique classes predicted:")
    print(f"  Method 1: {len(all_classes1)} classes")
    print(f"  Method 2: {len(all_classes2)} classes")
    print(f"  Common:   {len(all_classes1 & all_classes2)} classes")
    print(f"  Only in Method 1: {len(all_classes1 - all_classes2)} classes")
    print(f"  Only in Method 2: {len(all_classes2 - all_classes1)} classes")
    
    # Most common classes
    counter1 = Counter()
    counter2 = Counter()
    for sid in common_ids:
        counter1.update(pred1[sid])
        counter2.update(pred2[sid])
    
    print("\nTop-10 most predicted classes:")
    print("  Method 1:")
    for cls, count in counter1.most_common(10):
        print(f"    Class {cls}: {count} times")
    
    print("\n  Method 2:")
    for cls, count in counter2.most_common(10):
        print(f"    Class {cls}: {count} times")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare two submission files")
    parser.add_argument("--file1", type=str, required=True,
                        help="First submission file")
    parser.add_argument("--file2", type=str, required=True,
                        help="Second submission file")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample IDs to compare (default: first 10)")
    parser.add_argument("--all", action="store_true",
                        help="Compare all samples")
    
    args = parser.parse_args()
    
    sample_ids = None
    if args.samples:
        sample_ids = [int(x.strip()) for x in args.samples.split(',')]
    elif not args.all:
        # Will use first 10 by default
        pass
    
    compare_submissions(args.file1, args.file2, sample_ids)
