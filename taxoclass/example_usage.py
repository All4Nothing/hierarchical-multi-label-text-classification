"""
Example usage of TaxoClass components
"""
import torch
from transformers import BertTokenizer

from config import Config
from utils.hierarchy import TaxonomyHierarchy
from data.loader import DocumentCorpus
from models.similarity import DocumentClassSimilarity
from models.core_mining import CoreClassMiner
from models.classifier import TaxoClassifier
from utils.metrics import predict_top_k_classes


def example_prediction():
    """Example: Predict classes for new documents"""
    
    print("Loading hierarchy and model...")
    
    # Load hierarchy
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    # Load trained model
    model = TaxoClassifier(
        num_classes=hierarchy.num_classes,
        doc_encoder_name=Config.DOC_ENCODER_MODEL,
        embedding_dim=Config.EMBEDDING_DIM,
        gnn_hidden_dim=Config.GNN_HIDDEN_DIM,
        gnn_num_layers=Config.GNN_NUM_LAYERS
    )
    
    # Load model weights
    checkpoint = torch.load(f"{Config.MODEL_SAVE_DIR}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)
    
    # Get edge index
    edge_index = torch.LongTensor(
        hierarchy.get_edge_index(bidirectional=Config.GNN_BIDIRECTIONAL_EDGES)
    ).to(Config.DEVICE)
    
    # Example documents
    documents = [
        "This organic dark chocolate bar has a rich cocoa flavor with hints of vanilla.",
        "Educational STEM toy kit with building blocks for kids ages 5-10.",
        "Natural skin care moisturizer with vitamin E and aloe vera extract."
    ]
    
    print("\nPredicting classes for example documents...\n")
    
    # Predict
    results = predict_top_k_classes(
        model=model,
        documents=documents,
        tokenizer=tokenizer,
        edge_index=edge_index,
        hierarchy=hierarchy,
        device=Config.DEVICE,
        k=5
    )
    
    # Print results
    for i, (doc, predictions) in enumerate(zip(documents, results)):
        print(f"Document {i+1}: {doc[:80]}...")
        print("\nTop-5 Predicted Classes:")
        for j, (class_id, class_name, score) in enumerate(predictions):
            level = hierarchy.get_level(class_id)
            print(f"  {j+1}. {class_name} (Level: {level}, Score: {score:.4f})")
        print("\n" + "-"*80 + "\n")


def example_similarity_calculation():
    """Example: Calculate document-class similarities"""
    
    print("Loading data...")
    
    # Load hierarchy
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    # Load corpus
    corpus = DocumentCorpus(Config.TRAIN_CORPUS)
    documents = corpus.get_all_texts()[:100]  # Use first 100 documents
    
    # Calculate similarities
    print("\nCalculating similarities...")
    similarity_calculator = DocumentClassSimilarity(
        model_name=Config.SIMILARITY_MODEL,
        device=Config.DEVICE,
        batch_size=16
    )
    
    similarity_matrix = similarity_calculator.compute_similarity_matrix(
        documents=documents,
        class_names=hierarchy.id_to_name,
        use_cache=False
    )
    
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    
    # Show top-5 classes for first document
    doc_id = 0
    top_k = similarity_calculator.get_top_k_classes(doc_id, similarity_matrix, k=5)
    
    print(f"\nDocument: {documents[doc_id][:100]}...")
    print("\nTop-5 Similar Classes:")
    for class_id, score in top_k:
        class_name = hierarchy.id_to_name[class_id]
        print(f"  {class_name}: {score:.4f}")


def example_hierarchy_exploration():
    """Example: Explore taxonomy hierarchy"""
    
    print("Loading hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    print(f"\nTotal classes: {hierarchy.num_classes}")
    print(f"Max depth: {max(hierarchy.levels.values())}")
    
    # Get root nodes
    roots = hierarchy.get_roots()
    print(f"\nRoot nodes: {len(roots)}")
    for root_id in roots[:5]:
        root_name = hierarchy.id_to_name[root_id]
        num_children = len(hierarchy.get_children(root_id))
        print(f"  {root_name} ({num_children} children)")
    
    # Explore a specific class
    class_name = "toys_games"
    class_id = hierarchy.name_to_id.get(class_name)
    
    if class_id is not None:
        print(f"\nExploring class: {class_name}")
        print(f"  Level: {hierarchy.get_level(class_id)}")
        
        parents = hierarchy.get_parents(class_id)
        if parents:
            print(f"  Parents: {[hierarchy.id_to_name[p] for p in parents]}")
        
        children = hierarchy.get_children(class_id)
        if children:
            print(f"  Children ({len(children)}): {[hierarchy.id_to_name[c] for c in children[:5]]}...")
        
        ancestors = hierarchy.get_ancestors(class_id)
        print(f"  Total ancestors: {len(ancestors)}")
        
        descendants = hierarchy.get_descendants(class_id)
        print(f"  Total descendants: {len(descendants)}")


def example_core_class_analysis():
    """Example: Analyze core class mining results"""
    
    print("This example requires running Stages 1 and 2 first.")
    print("Please run main.py or execute similarity calculation and core mining.")
    
    # This would typically load cached results
    # For demonstration, we show the analysis structure
    
    print("\nCore Class Analysis includes:")
    print("  - Distribution across hierarchy levels")
    print("  - Top frequent core classes")
    print("  - Confidence score statistics")
    print("  - Coverage of taxonomy")


if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "TaxoClass Example Usage")
    print("="*80)
    
    print("\nAvailable examples:")
    print("1. Hierarchy Exploration")
    print("2. Similarity Calculation")
    print("3. Core Class Analysis")
    print("4. Prediction on New Documents")
    
    # Run hierarchy exploration (doesn't require trained model)
    print("\n" + "="*80)
    print("Running Example: Hierarchy Exploration")
    print("="*80)
    example_hierarchy_exploration()
    
    # Uncomment to run other examples:
    # example_similarity_calculation()
    # example_core_class_analysis()
    # example_prediction()  # Requires trained model

