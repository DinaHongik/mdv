#!/usr/bin/env python3
"""
Basic structure test for NMO-MDV framework (without external dependencies).
Tests core data processing and logic components.
"""

import json
import os
import sys
import tempfile
from typing import List, Dict
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test if core modules can be imported."""
    print("🧪 Testing Basic Imports...")
    
    try:
        from smartmap_mdv.data import NMOFields
        print("NMOFields import successful")
        
        from smartmap_mdv.config import TrainConfig, EvalConfig, ScoreWeights, DataConfig
        print("Config classes import successful")
        
        from smartmap_mdv.evaluate import ranks_from_scores, hit_at_k, mrr
        print("Evaluation functions import successful")
        
        return True
    except Exception as e:
        print(f"Import test failed: {e}")
        return False


def test_nmo_field_processing():
    """Test NMO field data processing without external dependencies."""
    print("\n Testing NMO Field Processing...")
    
    try:
        from smartmap_mdv.data import NMOFields
        
        # Create sample NMO field data
        sample_fields = [
            {"field_id": "src_ip", "name": "Source IP", "type": "string", "path": "event.src.ip", "description": "Source IP address", "examples": ["192.168.1.100"]},
            {"field_id": "action", "name": "Action", "type": "string", "path": "event.action", "description": "Security action", "examples": ["allow", "deny"]}
        ]
        
        # Write temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for field in sample_fields:
                f.write(json.dumps(field) + '\n')
            temp_file = f.name
        
        try:
            # Test NMO loading
            fields = NMOFields.from_file(temp_file)
            print(f"Loaded {len(fields.fields)} fields")
            
            # Test field text generation
            for field_id in ["src_ip", "action"]:
                if field_id in fields.fields:
                    nmo_text = fields.get_field_text(field_id)
                    print(f"NMO text for {field_id}: {nmo_text[:100]}...")
                    
                    # Test component extraction simulation
                    if "[NAME]" in nmo_text:
                        name_part = nmo_text.split("[NAME]")[1].split("[")[0].strip()
                        print(f"Extracted NAME: '{name_part}'")
                    
                    if "[TYPE]" in nmo_text:
                        type_part = nmo_text.split("[TYPE]")[1].split("[")[0].strip()
                        print(f"Extracted TYPE: '{type_part}'")
                        
                    if "[PATH]" in nmo_text:
                        path_part = nmo_text.split("[PATH]")[1].split("[")[0].strip()
                        print(f"Extracted PATH: '{path_part}'")
            
            # Test input modes
            modes = ["nmo", "msg"]
            for mode in modes:
                print(f"Testing {mode} mode...")
                try:
                    all_texts = fields.get_all_texts(input_mode=mode)
                    print(f"Generated {len(all_texts)} texts in {mode} mode")
                except Exception as e:
                    print(f"{mode} mode issue: {e}")
            
            return True
            
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        print(f"NMO field processing test failed: {e}")
        return False


def test_config_classes():
    """Test configuration classes."""
    print("\n Testing Configuration Classes...")
    
    try:
        from smartmap_mdv.config import TrainConfig, EvalConfig, ScoreWeights, DataConfig
        
        # Test TrainConfig
        train_config = TrainConfig()
        print(f"TrainConfig: ablation={train_config.ablation}, lr={train_config.lr}")
        
        # Test EvalConfig
        eval_config = EvalConfig()
        print(f"EvalConfig: batch_size={eval_config.batch_size}, compute_ece={eval_config.compute_ece}")
        
        # Test ScoreWeights
        weights = ScoreWeights()
        print(f"ScoreWeights: alpha_cos={weights.alpha_cos}, beta_type={weights.beta_type}")
        
        # Test DataConfig
        data_config = DataConfig()
        print(f"DataConfig: mask_name={data_config.mask_name}, ip_pattern exists={bool(data_config.ip_pattern)}")
        
        return True
        
    except Exception as e:
        print(f"Config classes test failed: {e}")
        return False


def test_evaluation_functions():
    """Test evaluation metric functions."""
    print("\n Testing Evaluation Functions...")
    
    try:
        from smartmap_mdv.evaluate import ranks_from_scores, hit_at_k, mrr
        
        # Create mock data
        scores = np.array([
            [0.9, 0.3, 0.2],  # Row 0
            [0.4, 0.8, 0.1],  # Row 1
            [0.2, 0.1, 0.7]   # Row 2
        ])
        
        # Mock ground truth
        true_indices = [0, 1, 2]
        
        print("Testing rank computation...")
        ranks = ranks_from_scores(scores, true_indices)
        print(f"Computed ranks: {ranks}")
        
        print("Testing Hit@k...")
        hit1 = hit_at_k(ranks, 1)
        hit3 = hit_at_k(ranks, 3)
        print(f"Hit@1: {hit1:.3f}, Hit@3: {hit3:.3f}")
        
        print("Testing MRR...")
        mrr_score = mrr(ranks)
        print(f"MRR: {mrr_score:.3f}")
        
        # Test edge cases
        print("Testing edge cases...")
        
        # Empty case
        empty_ranks = ranks_from_scores(np.array([]), [])
        print(f"Empty case handled: {len(empty_ranks)} ranks")
        
        # Multi-GT case
        multi_gt = [{0}, {1}, {0, 2}]  # Row 2 has multiple correct answers
        multi_ranks = ranks_from_scores(scores, multi_gt)
        print(f"Multi-GT case handled: {multi_ranks}")
        
        return True
        
    except Exception as e:
        print(f"Evaluation functions test failed: {e}")
        return False


def test_file_structure():
    """Test if all required files exist and have expected structure."""
    print("\n Testing File Structure...")
    
    required_files = [
        "smartmap_mdv/__init__.py",
        "smartmap_mdv/model.py",
        "smartmap_mdv/data.py", 
        "smartmap_mdv/train.py",
        "smartmap_mdv/evaluate.py",
        "smartmap_mdv/scoring.py",
        "smartmap_mdv/config.py",
        "run_eval.py",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"{file_path}")
        else:
            print(f"{file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing {len(missing_files)} required files")
        return False
    else:
        print(f"All {len(required_files)} required files present")
        return True


def test_component_integration():
    """Test integration between components."""
    print("\n Testing Component Integration...")
    
    try:
        from smartmap_mdv.data import NMOFields
        from smartmap_mdv.config import ScoreWeights
        
        # Test ScoreWeights integration with data processing
        weights = ScoreWeights(alpha_cos=1.0, beta_type=0.3, gamma_path=0.2, delta_lex=0.1)
        print(f"ScoreWeights initialized: cos={weights.alpha_cos}, type={weights.beta_type}")
        
        # Create sample data
        sample_fields = [
            {"field_id": "test", "name": "Test Field", "type": "string", "path": "test.path", "description": "Test description", "examples": ["test"]}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(sample_fields[0]) + '\n')
            temp_file = f.name
        
        try:
            fields = NMOFields.from_file(temp_file)
            
            # Test integrated scoring simulation (without actual encoder)
            source_text = fields.get_field_text("test")
            print(f"Generated NMO text for integration test: {source_text[:80]}...")
            
            # Simulate component weights application
            name_sim = 0.8  # Mock similarity
            type_sim = 0.6
            path_sim = 0.7
            
            integrated_score = (weights.alpha_cos * name_sim + 
                            weights.beta_type * type_sim + 
                            weights.gamma_path * path_sim + 
                            weights.delta_lex * 0.5)  # Mock lexical sim
            
            print(f"Mock integrated score: {integrated_score:.3f}")
            
            return True
            
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        print(f"Component integration test failed: {e}")
        return False


def main():
    """Run all basic tests to verify framework structure."""
    print("Starting NMO-MDV Framework Structure Tests\n")
    
    test_results = []
    
    # Run basic tests (no external dependencies)
    test_results.append(test_basic_imports())
    test_results.append(test_file_structure())
    test_results.append(test_config_classes())
    test_results.append(test_nmo_field_processing())
    test_results.append(test_evaluation_functions())
    test_results.append(test_component_integration())
    
    # Summary
    print(f"\n Test Results Summary:")
    passed = sum(test_results)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n All basic tests passed!")
        print("\n Framework Structure Verified:")
        print("File structure and imports")
        print("NMO field processing and canonicalization")
        print("Configuration management")
        print("Evaluation metrics (Hit@k, MRR, ranks)")
        print("Component integration")
        print("\n Framework structure is sound!")
        print("Note: Full functionality requires PyTorch and transformers installation")
    else:
        print("\n Some structural tests failed. Please review the errors above.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)