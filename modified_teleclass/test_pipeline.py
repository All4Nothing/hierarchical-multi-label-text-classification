"""
Quick test script to validate pipeline components.
Run this before executing the full pipeline to catch any issues early.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import sys
import torch

def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    try:
        import transformers
        import sentence_transformers
        import networkx
        import pandas
        import numpy
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Install with: pip install -r requirements_teleclass.txt")
        return False

def test_data_files():
    """Test that all required data files exist."""
    print("\nTesting data files...")
    data_dir = "../Amazon_products"
    
    required_files = [
        "train/train_corpus.txt",
        "test/test_corpus.txt",
        "class_hierarchy.txt",
        "class_related_keywords.txt",
        "classes.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(data_dir, file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA is available")
        print(f"  Devices: {device_count}")
        print(f"  Device 0: {device_name}")
        return True
    else:
        print("⚠ CUDA not available (will use CPU)")
        return False

def test_pipeline_import():
    """Test that the pipeline can be imported."""
    print("\nTesting pipeline import...")
    try:
        from pipeline_teleclass import TELEClassPipeline, DataLoader
        print("✓ Pipeline imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import pipeline: {e}")
        return False

def test_data_loader():
    """Test DataLoader on actual files."""
    print("\nTesting DataLoader...")
    try:
        from pipeline_teleclass import DataLoader
        
        loader = DataLoader("../Amazon_products")
        loader.load_all()
        
        print(f"✓ Loaded {len(loader.train_corpus)} train documents")
        print(f"✓ Loaded {len(loader.test_corpus)} test documents")
        print(f"✓ Loaded {len(loader.all_classes)} classes")
        print(f"✓ Loaded {len(loader.class_keywords)} class keywords")
        print(f"✓ Hierarchy has {loader.hierarchy_graph.number_of_edges()} edges")
        
        return True
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TELEClass Pipeline - Pre-flight Checks")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Data Files": test_data_files(),
        "CUDA": test_cuda(),
        "Pipeline Import": test_pipeline_import(),
        "DataLoader": test_data_loader()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run pipeline.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before running pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
