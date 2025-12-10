"""
Visualize similarity matrix for TaxoClass framework
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

from config import Config
from utils.hierarchy import TaxonomyHierarchy


def load_similarity_matrix(filepath):
    """Load similarity matrix from npz file"""
    print(f"Loading similarity matrix from {filepath}...")
    data = np.load(filepath)
    
    # Handle different possible keys
    if 'similarity_matrix' in data:
        matrix = data['similarity_matrix']
    elif 'matrix' in data:
        matrix = data['matrix']
    else:
        # Assume first array is the matrix
        keys = list(data.keys())
        matrix = data[keys[0]]
    
    print(f"✅ Loaded matrix shape: {matrix.shape}")
    print(f"   Data type: {matrix.dtype}")
    print(f"   Value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"   Mean: {matrix.mean():.4f}, Std: {matrix.std():.4f}")
    
    return matrix


def plot_matrix_overview(matrix, output_dir="outputs/visualizations"):
    """Plot overall similarity matrix overview"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample if too large
    max_size = 1000
    if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
        print(f"Matrix too large ({matrix.shape}), sampling...")
        doc_sample = np.random.choice(matrix.shape[0], min(max_size, matrix.shape[0]), replace=False)
        class_sample = np.random.choice(matrix.shape[1], min(max_size, matrix.shape[1]), replace=False)
        matrix_sample = matrix[np.ix_(doc_sample, class_sample)]
    else:
        matrix_sample = matrix
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Full matrix heatmap (sampled)
    ax = axes[0, 0]
    sns.heatmap(matrix_sample, cmap='viridis', ax=ax, cbar=True, 
                xticklabels=False, yticklabels=False)
    ax.set_title(f'Similarity Matrix Heatmap (Sampled: {matrix_sample.shape})', fontsize=12)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Documents')
    
    # 2. Distribution of similarity scores
    ax = axes[0, 1]
    ax.hist(matrix.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Similarity Scores')
    ax.axvline(matrix.mean(), color='r', linestyle='--', label=f'Mean: {matrix.mean():.4f}')
    ax.legend()
    
    # 3. Per-document max similarity
    ax = axes[1, 0]
    doc_max = matrix.max(axis=1)
    ax.hist(doc_max, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Max Similarity per Document')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Max Similarity Distribution\n(Mean: {doc_max.mean():.4f})')
    
    # 4. Per-class mean similarity
    ax = axes[1, 1]
    class_mean = matrix.mean(axis=0)
    ax.hist(class_mean, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mean Similarity per Class')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Mean Similarity per Class\n(Mean: {class_mean.mean():.4f})')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'similarity_matrix_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_class_statistics(matrix, hierarchy, output_dir="outputs/visualizations"):
    """Plot statistics by class"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_classes = matrix.shape[1]
    class_stats = []
    
    for class_id in range(num_classes):
        if class_id not in hierarchy.id_to_name:
            continue
        
        class_similarities = matrix[:, class_id]
        class_stats.append({
            'class_id': class_id,
            'class_name': hierarchy.id_to_name[class_id],
            'level': hierarchy.get_level(class_id),
            'mean': class_similarities.mean(),
            'std': class_similarities.std(),
            'max': class_similarities.max(),
            'min': class_similarities.min(),
            'median': np.median(class_similarities),
            'above_threshold_05': (class_similarities > 0.5).sum(),
            'above_threshold_07': (class_similarities > 0.7).sum(),
        })
    
    df = pd.DataFrame(class_stats)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Top classes by mean similarity
    ax = axes[0, 0]
    top_classes = df.nlargest(20, 'mean')
    ax.barh(range(len(top_classes)), top_classes['mean'].values)
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels([f"{row['class_id']}: {row['class_name'][:30]}" 
                        for _, row in top_classes.iterrows()], fontsize=8)
    ax.set_xlabel('Mean Similarity')
    ax.set_title('Top 20 Classes by Mean Similarity')
    ax.invert_yaxis()
    
    # 2. Similarity by hierarchy level
    ax = axes[0, 1]
    level_stats = df.groupby('level')['mean'].agg(['mean', 'std', 'count'])
    ax.bar(level_stats.index, level_stats['mean'], yerr=level_stats['std'], 
           capsize=5, alpha=0.7)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Mean Similarity by Hierarchy Level')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Distribution of mean similarities
    ax = axes[0, 2]
    ax.hist(df['mean'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mean Similarity')
    ax.set_ylabel('Number of Classes')
    ax.set_title('Distribution of Mean Similarities')
    ax.axvline(df['mean'].mean(), color='r', linestyle='--', 
               label=f'Overall Mean: {df["mean"].mean():.4f}')
    ax.legend()
    
    # 4. Classes with high max similarity
    ax = axes[1, 0]
    high_max = df.nlargest(20, 'max')
    ax.barh(range(len(high_max)), high_max['max'].values)
    ax.set_yticks(range(len(high_max)))
    ax.set_yticklabels([f"{row['class_id']}: {row['class_name'][:30]}" 
                        for _, row in high_max.iterrows()], fontsize=8)
    ax.set_xlabel('Max Similarity')
    ax.set_title('Top 20 Classes by Max Similarity')
    ax.invert_yaxis()
    
    # 5. Classes with many high-similarity documents
    ax = axes[1, 1]
    high_count = df.nlargest(20, 'above_threshold_05')
    ax.barh(range(len(high_count)), high_count['above_threshold_05'].values)
    ax.set_yticks(range(len(high_count)))
    ax.set_yticklabels([f"{row['class_id']}: {row['class_name'][:30]}" 
                        for _, row in high_count.iterrows()], fontsize=8)
    ax.set_xlabel('Number of Documents (sim > 0.5)')
    ax.set_title('Top 20 Classes by High-Similarity Count')
    ax.invert_yaxis()
    
    # 6. Similarity variance by level
    ax = axes[1, 2]
    level_variance = df.groupby('level')['std'].mean()
    ax.bar(level_variance.index, level_variance.values, alpha=0.7)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Mean Std Dev')
    ax.set_title('Similarity Variance by Level')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'class_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # Save statistics to CSV
    csv_path = os.path.join(output_dir, 'class_statistics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")


def plot_document_statistics(matrix, output_dir="outputs/visualizations", sample_size=1000):
    """Plot statistics by document"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_docs = matrix.shape[0]
    
    # Sample documents if too many
    if num_docs > sample_size:
        doc_indices = np.random.choice(num_docs, sample_size, replace=False)
        matrix_sample = matrix[doc_indices]
    else:
        matrix_sample = matrix
        doc_indices = np.arange(num_docs)
    
    # Calculate statistics
    doc_max = matrix_sample.max(axis=1)
    doc_mean = matrix_sample.mean(axis=1)
    doc_std = matrix_sample.std(axis=1)
    doc_top_k_count = (matrix_sample > 0.5).sum(axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Max similarity per document
    ax = axes[0, 0]
    ax.hist(doc_max, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Max Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Max Similarity per Document\n(Mean: {doc_max.mean():.4f})')
    ax.axvline(doc_max.mean(), color='r', linestyle='--', label='Mean')
    ax.legend()
    
    # 2. Mean similarity per document
    ax = axes[0, 1]
    ax.hist(doc_mean, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mean Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Mean Similarity per Document\n(Mean: {doc_mean.mean():.4f})')
    ax.axvline(doc_mean.mean(), color='r', linestyle='--', label='Mean')
    ax.legend()
    
    # 3. Number of high-similarity classes per document
    ax = axes[1, 0]
    ax.hist(doc_top_k_count, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Classes (sim > 0.5)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'High-Similarity Classes per Document\n(Mean: {doc_top_k_count.mean():.1f})')
    ax.axvline(doc_top_k_count.mean(), color='r', linestyle='--', label='Mean')
    ax.legend()
    
    # 4. Scatter: max vs mean similarity
    ax = axes[1, 1]
    ax.scatter(doc_mean, doc_max, alpha=0.3, s=10)
    ax.set_xlabel('Mean Similarity')
    ax.set_ylabel('Max Similarity')
    ax.set_title('Max vs Mean Similarity per Document')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'document_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_hierarchical_analysis(matrix, hierarchy, output_dir="outputs/visualizations"):
    """Plot hierarchical structure analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_classes = matrix.shape[1]
    
    # Group classes by level
    level_classes = defaultdict(list)
    for class_id in range(num_classes):
        if class_id in hierarchy.id_to_name:
            level = hierarchy.get_level(class_id)
            level_classes[level].append(class_id)
    
    # Calculate statistics per level
    level_stats = []
    for level in sorted(level_classes.keys()):
        class_ids = level_classes[level]
        level_matrix = matrix[:, class_ids]
        
        level_stats.append({
            'level': level,
            'num_classes': len(class_ids),
            'mean_similarity': level_matrix.mean(),
            'std_similarity': level_matrix.std(),
            'max_similarity': level_matrix.max(),
            'min_similarity': level_matrix.min(),
        })
    
    df_levels = pd.DataFrame(level_stats)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Number of classes per level
    ax = axes[0, 0]
    ax.bar(df_levels['level'], df_levels['num_classes'], alpha=0.7)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Number of Classes')
    ax.set_title('Class Distribution by Level')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Mean similarity by level
    ax = axes[0, 1]
    ax.bar(df_levels['level'], df_levels['mean_similarity'], 
           yerr=df_levels['std_similarity'], capsize=5, alpha=0.7)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Mean Similarity by Hierarchy Level')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Max similarity by level
    ax = axes[1, 0]
    ax.bar(df_levels['level'], df_levels['max_similarity'], alpha=0.7)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Max Similarity')
    ax.set_title('Max Similarity by Hierarchy Level')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Similarity range by level
    ax = axes[1, 1]
    ax.plot(df_levels['level'], df_levels['max_similarity'], 'o-', label='Max', linewidth=2)
    ax.plot(df_levels['level'], df_levels['mean_similarity'], 's-', label='Mean', linewidth=2)
    ax.plot(df_levels['level'], df_levels['min_similarity'], '^-', label='Min', linewidth=2)
    ax.fill_between(df_levels['level'], df_levels['min_similarity'], 
                    df_levels['max_similarity'], alpha=0.2)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity Range by Level')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hierarchical_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_top_k_analysis(matrix, hierarchy, k=10, output_dir="outputs/visualizations"):
    """Analyze top-K classes for each document"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_docs = matrix.shape[0]
    num_classes = matrix.shape[1]
    
    # Count how many times each class appears in top-K
    class_top_k_count = np.zeros(num_classes)
    top_k_classes_per_doc = []
    
    for doc_idx in range(num_docs):
        top_k_indices = np.argsort(matrix[doc_idx])[-k:][::-1]
        top_k_classes_per_doc.append(top_k_indices)
        class_top_k_count[top_k_indices] += 1
    
    # Get top classes
    top_classes = np.argsort(class_top_k_count)[-20:][::-1]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Most frequent top-K classes
    ax = axes[0, 0]
    top_counts = class_top_k_count[top_classes]
    class_names = [hierarchy.id_to_name.get(cid, f"Class_{cid}")[:30] 
                   for cid in top_classes]
    ax.barh(range(len(top_classes)), top_counts)
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels([f"{cid}: {name}" for cid, name in zip(top_classes, class_names)], 
                       fontsize=8)
    ax.set_xlabel(f'Number of Documents (in Top-{k})')
    ax.set_title(f'Most Frequent Classes in Top-{k}')
    ax.invert_yaxis()
    
    # 2. Distribution of top-K counts
    ax = axes[0, 1]
    ax.hist(class_top_k_count[class_top_k_count > 0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'Frequency in Top-{k}')
    ax.set_ylabel('Number of Classes')
    ax.set_title(f'Distribution of Top-{k} Frequencies')
    
    # 3. Average similarity of top-K classes
    ax = axes[1, 0]
    top_k_avg_sim = []
    for doc_idx in range(min(1000, num_docs)):  # Sample for speed
        top_k_indices = np.argsort(matrix[doc_idx])[-k:][::-1]
        top_k_avg_sim.append(matrix[doc_idx, top_k_indices].mean())
    
    ax.hist(top_k_avg_sim, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Average Similarity of Top-K')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Top-{k} Average Similarity\n(Mean: {np.mean(top_k_avg_sim):.4f})')
    
    # 4. Top-K diversity (number of unique classes)
    ax = axes[1, 1]
    num_unique_in_topk = (class_top_k_count > 0).sum()
    ax.bar(['In Top-K', 'Not in Top-K'], 
           [num_unique_in_topk, num_classes - num_unique_in_topk], alpha=0.7)
    ax.set_ylabel('Number of Classes')
    ax.set_title(f'Class Coverage in Top-{k}\n({num_unique_in_topk}/{num_classes} classes)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'top_{k}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize similarity matrix for TaxoClass"
    )
    parser.add_argument("--matrix_file", type=str, 
                       default="outputs/similarity_matrix_all.npz",
                       help="Path to similarity matrix npz file")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--all", action="store_true",
                       help="Generate all visualizations")
    parser.add_argument("--overview", action="store_true",
                       help="Generate matrix overview")
    parser.add_argument("--class_stats", action="store_true",
                       help="Generate class statistics")
    parser.add_argument("--doc_stats", action="store_true",
                       help="Generate document statistics")
    parser.add_argument("--hierarchical", action="store_true",
                       help="Generate hierarchical analysis")
    parser.add_argument("--top_k", type=int, default=10,
                       help="K for top-K analysis")
    
    args = parser.parse_args()
    
    # Load similarity matrix
    matrix = load_similarity_matrix(args.matrix_file)
    
    # Load hierarchy
    print("\nLoading hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    print(f"✅ Loaded: {hierarchy.num_classes} classes")
    
    # Generate visualizations
    if args.all or args.overview:
        print("\n" + "="*80)
        print("GENERATING MATRIX OVERVIEW")
        print("="*80)
        plot_matrix_overview(matrix, args.output_dir)
    
    if args.all or args.class_stats:
        print("\n" + "="*80)
        print("GENERATING CLASS STATISTICS")
        print("="*80)
        plot_class_statistics(matrix, hierarchy, args.output_dir)
    
    if args.all or args.doc_stats:
        print("\n" + "="*80)
        print("GENERATING DOCUMENT STATISTICS")
        print("="*80)
        plot_document_statistics(matrix, args.output_dir)
    
    if args.all or args.hierarchical:
        print("\n" + "="*80)
        print("GENERATING HIERARCHICAL ANALYSIS")
        print("="*80)
        plot_hierarchical_analysis(matrix, hierarchy, args.output_dir)
    
    if args.all:
        print("\n" + "="*80)
        print("GENERATING TOP-K ANALYSIS")
        print("="*80)
        plot_top_k_analysis(matrix, hierarchy, k=args.top_k, output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
