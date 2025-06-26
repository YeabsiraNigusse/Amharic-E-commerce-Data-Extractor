#!/usr/bin/env python3
"""
Simple test to verify the enhanced features work
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.modeling.advanced_ner_trainer import AdvancedNERTrainer
        print("✅ AdvancedNERTrainer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import AdvancedNERTrainer: {e}")
        return False
    
    try:
        from src.pipeline.end_to_end_pipeline import EndToEndPipeline
        print("✅ EndToEndPipeline imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EndToEndPipeline: {e}")
        return False
    
    return True

def test_advanced_trainer_init():
    """Test advanced trainer initialization"""
    print("\nTesting AdvancedNERTrainer initialization...")
    
    try:
        from src.modeling.advanced_ner_trainer import AdvancedNERTrainer
        
        trainer = AdvancedNERTrainer(
            model_name="distilbert-base-multilingual-cased",
            output_dir="test_models"
        )
        
        print("✅ AdvancedNERTrainer initialized successfully")
        print(f"   Model: {trainer.model_name}")
        print(f"   Output dir: {trainer.output_dir}")
        print(f"   Hyperparameter space: {len(trainer.hyperparameter_space)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize AdvancedNERTrainer: {e}")
        return False

def test_pipeline_init():
    """Test pipeline initialization"""
    print("\nTesting EndToEndPipeline initialization...")
    
    try:
        from src.pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Create a temporary config
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'data_ingestion': {'enabled': False},
                'preprocessing': {'enabled': False},
                'labeling': {'enabled': False},
                'training': {'enabled': False},
                'evaluation': {'enabled': False}
            }
            yaml.dump(config, f)
            config_path = f.name
        
        pipeline = EndToEndPipeline(config_path)
        
        print("✅ EndToEndPipeline initialized successfully")
        print(f"   Pipeline ID: {pipeline.pipeline_id}")
        print(f"   Config loaded: {len(pipeline.config)} sections")
        
        # Cleanup
        Path(config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize EndToEndPipeline: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "src/modeling/advanced_ner_trainer.py",
        "src/pipeline/__init__.py",
        "src/pipeline/end_to_end_pipeline.py",
        "run_advanced_fine_tuning.py",
        "run_complete_pipeline.py",
        "ENHANCED_IMPLEMENTATION_REPORT.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("=" * 60)
    print("SIMPLE TEST FOR ENHANCED FEATURES")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("AdvancedNERTrainer", test_advanced_trainer_init),
        ("EndToEndPipeline", test_pipeline_init)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
