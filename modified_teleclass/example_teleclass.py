"""
Example usage of the TELEClass pipeline.
Demonstrates how to use individual components and customize parameters.
"""

from pipeline_teleclass import (
    TELEClassPipeline,
    DataLoader,
    ClassRepresentationModule,
    IterativePseudoLabeler,
    AugmentationModule,
    HierarchyExpander,
    BERTClassifierTrainer,
    InferenceModule,
    set_seed
)
import torch
import os


def example_1_basic_pipeline():
    """
    Example 1: Run the complete pipeline with default settings.
    This is the simplest way to use the pipeline.
    """
    print("="*80)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("="*80)
    
    pipeline = TELEClassPipeline(
        data_dir="Amazon_products",
        output_dir="outputs",
        seed=42
    )
    pipeline.run()


def example_2_custom_parameters():
    """
    Example 2: Run pipeline with custom parameters.
    Shows how to adjust hyperparameters for better performance.
    """
    print("="*80)
    print("EXAMPLE 2: Custom Parameters")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load data
    data_loader = DataLoader("Amazon_products")
    data_loader.load_all()
    
    # Phase 1: Class Representation
    class_repr = ClassRepresentationModule(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    class_descriptions = class_repr.create_class_descriptions(data_loader.class_keywords)
    class_embeddings = class_repr.encode_classes(class_descriptions, data_loader.all_classes)
    doc_embeddings = class_repr.encode_documents(data_loader.all_corpus, batch_size=64)
    
    # Phase 2: Iterative Refinement with custom parameters
    labeler = IterativePseudoLabeler()
    refined_embeddings, similarity = labeler.refine_class_embeddings(
        doc_embeddings,
        class_embeddings,
        num_iterations=5,  # More iterations
        top_n_reliable=30,  # More documents for centroid
        initial_top_k=100
    )
    
    pseudo_labels, pseudo_scores = labeler.assign_labels_with_gap(
        similarity,
        min_labels=3,  # At least 3 labels per document
        max_gap_search=10  # Look further for gaps
    )
    
    print(f"Average labels per document: {sum(len(l) for l in pseudo_labels) / len(pseudo_labels):.2f}")


def example_3_incremental_phases():
    """
    Example 3: Run phases incrementally and save intermediate results.
    Useful for debugging and experimentation.
    """
    print("="*80)
    print("EXAMPLE 3: Incremental Phase Execution")
    print("="*80)
    
    set_seed(42)
    output_dir = "outputs_incremental"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    data_loader = DataLoader("Amazon_products")
    data_loader.load_all()
    
    # Phase 1
    print("\n[2/6] Phase 1: Encoding...")
    class_repr = ClassRepresentationModule()
    class_descriptions = class_repr.create_class_descriptions(data_loader.class_keywords)
    class_embeddings = class_repr.encode_classes(class_descriptions, data_loader.all_classes)
    doc_embeddings = class_repr.encode_documents(data_loader.all_corpus)
    
    # Save embeddings
    torch.save({
        'class_embeddings': class_embeddings,
        'doc_embeddings': doc_embeddings
    }, os.path.join(output_dir, "embeddings.pt"))
    print(f"  Saved embeddings to {output_dir}/embeddings.pt")
    
    # Phase 2
    print("\n[3/6] Phase 2: Pseudo-labeling...")
    labeler = IterativePseudoLabeler()
    refined_embeddings, similarity = labeler.refine_class_embeddings(
        doc_embeddings, class_embeddings
    )
    pseudo_labels, pseudo_scores = labeler.assign_labels_with_gap(similarity)
    
    # Save pseudo-labels
    torch.save({
        'pseudo_labels': pseudo_labels,
        'pseudo_scores': pseudo_scores,
        'similarity': similarity
    }, os.path.join(output_dir, "pseudo_labels.pt"))
    print(f"  Saved pseudo-labels to {output_dir}/pseudo_labels.pt")
    
    # Phase 3
    print("\n[4/6] Phase 3: Augmentation...")
    aug_module = AugmentationModule(data_loader)
    starved_classes = aug_module.identify_starved_classes(
        pseudo_labels, data_loader.train_indices
    )
    print(f"  Found {len(starved_classes)} starved classes")
    
    # Phase 4
    print("\n[5/6] Phase 4: Hierarchy expansion...")
    expander = HierarchyExpander(data_loader.hierarchy_graph, data_loader.class_to_idx)
    expanded_labels = expander.expand_labels_with_hierarchy(pseudo_labels)
    
    # Save expanded labels
    torch.save({
        'expanded_labels': expanded_labels
    }, os.path.join(output_dir, "expanded_labels.pt"))
    print(f"  Saved expanded labels to {output_dir}/expanded_labels.pt")
    
    print("\n[6/6] Complete! Intermediate results saved to:", output_dir)


def example_4_inference_only():
    """
    Example 4: Run inference with a pre-trained model.
    Assumes model is already trained and saved.
    """
    print("="*80)
    print("EXAMPLE 4: Inference Only")
    print("="*80)
    
    # Load data
    data_loader = DataLoader("Amazon_products")
    data_loader.load_all()
    
    # Load trained model
    model_path = "outputs/models/best_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using example_1_basic_pipeline()")
        return
    
    # Run inference
    inference = InferenceModule(model_path=model_path)
    
    test_predictions = inference.predict(
        data_loader.test_corpus,
        batch_size=32,
        threshold=0.3  # Lower threshold for more labels
    )
    
    # Apply hierarchy expansion
    expander = HierarchyExpander(data_loader.hierarchy_graph, data_loader.class_to_idx)
    test_predictions_expanded = expander.expand_labels_with_hierarchy(test_predictions)
    
    # Generate submission
    inference.generate_submission(
        test_predictions_expanded,
        data_loader.idx_to_class,
        output_path="outputs/submission_custom.csv"
    )


def example_5_analyze_results():
    """
    Example 5: Analyze pseudo-labels and intermediate results.
    """
    print("="*80)
    print("EXAMPLE 5: Result Analysis")
    print("="*80)
    
    import numpy as np
    
    # Load intermediate results
    results_path = "outputs/intermediate/phase2_outputs.pt"
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        print("Please run the pipeline first")
        return
    
    results = torch.load(results_path)
    pseudo_labels = results['pseudo_labels']
    pseudo_scores = results['pseudo_scores']
    
    # Analyze label distribution
    label_counts = [len(labels) for labels in pseudo_labels]
    
    print("\nLabel Distribution Statistics:")
    print(f"  Mean: {np.mean(label_counts):.2f}")
    print(f"  Median: {np.median(label_counts):.2f}")
    print(f"  Min: {np.min(label_counts)}")
    print(f"  Max: {np.max(label_counts)}")
    print(f"  Std: {np.std(label_counts):.2f}")
    
    # Analyze score distribution
    all_scores = [score for scores in pseudo_scores for score in scores]
    
    print("\nScore Distribution Statistics:")
    print(f"  Mean: {np.mean(all_scores):.4f}")
    print(f"  Median: {np.median(all_scores):.4f}")
    print(f"  Min: {np.min(all_scores):.4f}")
    print(f"  Max: {np.max(all_scores):.4f}")
    
    # Class frequency analysis
    class_counts = {}
    for labels in pseudo_labels:
        for class_idx in labels:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
    
    print("\nClass Frequency:")
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Most frequent: {max(class_counts.values())} documents")
    print(f"  Least frequent: {min(class_counts.values())} documents")
    print(f"  Average: {np.mean(list(class_counts.values())):.2f} documents per class")


def example_6_train_with_validation():
    """
    Example 6: Train classifier with validation split.
    Uses a portion of training data for validation.
    """
    print("="*80)
    print("EXAMPLE 6: Training with Validation")
    print("="*80)
    
    set_seed(42)
    
    # Load data and generate pseudo-labels (simplified)
    data_loader = DataLoader("Amazon_products")
    data_loader.load_all()
    
    # Assume we have pseudo-labels (load from previous run)
    results_path = "outputs/intermediate/phase2_outputs.pt"
    if not os.path.exists(results_path):
        print("Please run the pipeline first to generate pseudo-labels")
        return
    
    results = torch.load(results_path)
    pseudo_labels = results['pseudo_labels']
    
    # Split train data into train/val
    from sklearn.model_selection import train_test_split
    
    train_texts = data_loader.train_corpus
    train_labels = [pseudo_labels[i] for i in data_loader.train_indices]
    
    train_texts_split, val_texts_split, train_labels_split, val_labels_split = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Train size: {len(train_texts_split)}")
    print(f"Val size: {len(val_texts_split)}")
    
    # Train with validation
    trainer = BERTClassifierTrainer(num_classes=len(data_loader.all_classes))
    trainer.prepare_data(
        train_texts_split, train_labels_split,
        val_texts_split, val_labels_split,
        batch_size=16
    )
    
    trainer.train(
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="outputs/models_with_val"
    )


def main():
    """
    Main function - choose which example to run.
    """
    import sys
    
    examples = {
        "1": ("Basic Pipeline", example_1_basic_pipeline),
        "2": ("Custom Parameters", example_2_custom_parameters),
        "3": ("Incremental Phases", example_3_incremental_phases),
        "4": ("Inference Only", example_4_inference_only),
        "5": ("Analyze Results", example_5_analyze_results),
        "6": ("Train with Validation", example_6_train_with_validation),
    }
    
    print("TELEClass Pipeline - Example Usage")
    print("="*80)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("\nUsage:")
    print("  python example_teleclass.py [example_number]")
    print("  python example_teleclass.py 1")
    print("\nOr run all examples:")
    print("  python example_teleclass.py all")
    print()
    
    if len(sys.argv) < 2:
        print("Please specify an example number (1-6) or 'all'")
        sys.exit(1)
    
    choice = sys.argv[1]
    
    if choice == "all":
        for key, (name, func) in examples.items():
            print(f"\n{'='*80}")
            print(f"Running Example {key}: {name}")
            print('='*80)
            try:
                func()
            except Exception as e:
                print(f"Error in example {key}: {e}")
                import traceback
                traceback.print_exc()
    elif choice in examples:
        name, func = examples[choice]
        func()
    else:
        print(f"Invalid choice: {choice}")
        print("Please choose 1-6 or 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
