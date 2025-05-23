#!/usr/bin/env python3
"""
Test script for Qwen3 model loading fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_qwen3_config_fix():
    """Test the Qwen3 configuration fix."""
    from verl.verl.utils.hf_tokenizer import fix_qwen3_config
    from transformers import AutoConfig
    
    print("Testing Qwen3 configuration fix...")
    
    # Test with a mock config that has the issue
    class MockConfig:
        def __init__(self):
            self.model_type = 'qwen3'
            self._parallel_styles = None
            self.parallel_styles = None
            self.rope_scaling = {'type': None}
    
    mock_config = MockConfig()
    print(f"Before fix: _parallel_styles={mock_config._parallel_styles}, parallel_styles={mock_config.parallel_styles}")
    
    # Apply the fix
    fixed_config = fix_qwen3_config(mock_config)
    
    print(f"After fix: _parallel_styles={fixed_config._parallel_styles}, parallel_styles={fixed_config.parallel_styles}")
    print(f"Rope scaling type: {fixed_config.rope_scaling['type']}")
    
    # Check that the fix worked
    assert fixed_config._parallel_styles == []
    assert fixed_config.parallel_styles == []
    assert fixed_config.rope_scaling['type'] == 'default'
    
    print("✅ Qwen3 configuration fix test passed!")

def test_model_loading():
    """Test actual model loading with the fix."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        from verl.verl.utils.hf_tokenizer import fix_qwen3_config
        import warnings
        
        # Test with Qwen/Qwen3-4B model
        model_path = "Qwen/Qwen3-4B"
        
        print(f"Loading configuration for {model_path}...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Apply our fix
        config = fix_qwen3_config(config)
        
        print(f"Configuration fixed. Model type: {getattr(config, 'model_type', 'unknown')}")
        
        # Now try to load the model with the fixed config
        print("Attempting to load model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype='auto',
                trust_remote_code=True,
                device_map='cpu'  # Use CPU to avoid GPU memory issues
            )
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Running Qwen3 fix tests...")
    
    # Test the configuration fix
    test_qwen3_config_fix()
    
    # Test actual model loading
    if test_model_loading():
        print("\n🎉 All tests passed! The fix should work.")
    else:
        print("\n❌ Model loading test failed. The fix may need more work.") 