"""
Quick test script to verify installation and basic functionality
"""
import sys
import torch
import numpy as np

def check_imports():
    """Check if all required packages are installed"""
    print("Checking imports...")
    
    packages = {
        'torch': torch,
        'numpy': np,
        'transformers': None,
        'sklearn': None,
        'tqdm': None,
        'networkx': None,
        'pandas': None
    }
    
    failed = []
    
    for package_name, package in packages.items():
        try:
            if package is None:
                __import__(package_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT FOUND")
            failed.append(package_name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠️  CUDA not available, will use CPU")
    
    return True


def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    from config import Config
    import os
    
    files_to_check = [
        Config.CLASSES_FILE,
        Config.HIERARCHY_FILE,
        Config.TRAIN_CORPUS,
        Config.TEST_CORPUS
    ]
    
    all_exist = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some data files are missing!")
        print("Please check the DATA_DIR path in config.py")
        return False
    
    print("\n✅ All data files found!")
    return True


def test_hierarchy_loading():
    """Test loading taxonomy hierarchy"""
    print("\nTesting hierarchy loading...")
    
    try:
        from config import Config
        from utils.hierarchy import TaxonomyHierarchy
        
        hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
        
        print(f"  ✓ Loaded {hierarchy.num_classes} classes")
        print(f"  ✓ Max depth: {max(hierarchy.levels.values())}")
        print(f"  ✓ Root nodes: {len(hierarchy.get_roots())}")
        
        print("\n✅ Hierarchy loading successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading hierarchy: {e}")
        return False


def test_corpus_loading():
    """Test loading document corpus"""
    print("\nTesting corpus loading...")
    
    try:
        from config import Config
        from data.loader import DocumentCorpus
        
        corpus = DocumentCorpus(Config.TRAIN_CORPUS)
        
        print(f"  ✓ Loaded {len(corpus)} documents")
        print(f"  ✓ Sample document: {corpus.get_text(0)[:80]}...")
        
        print("\n✅ Corpus loading successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading corpus: {e}")
        return False


def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    
    try:
        from models.classifier import TaxoClassifier
        
        model = TaxoClassifier(
            num_classes=100,
            embedding_dim=768,
            gnn_hidden_dim=256,
            gnn_num_layers=2
        )
        
        print(f"  ✓ Model initialized")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n✅ Model initialization successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing model: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print(" "*15 + "TaxoClass Quick Test")
    print("="*60)
    
    tests = [
        check_imports,
        check_cuda,
        check_data_files,
        test_hierarchy_loading,
        test_corpus_loading,
        test_model_initialization
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append(False)
        
        print()
    
    # Summary
    print("="*60)
    print(" "*20 + "Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ All tests passed! You can run main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

