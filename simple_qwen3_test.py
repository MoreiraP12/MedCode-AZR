#!/usr/bin/env python3
"""
Simple test script for Qwen3 model loading fix.
This test doesn't require the full verl environment.
"""

def fix_qwen3_config(config):
    """
    Fix Qwen3 configuration issues related to parallel_styles being None.
    
    Args:
        config: The model configuration object
        
    Returns:
        Fixed configuration object
    """
    # Check if this is a Qwen3 model and fix the parallel_styles issue
    if hasattr(config, 'model_type') and 'qwen3' in str(config.model_type).lower():
        # Fix the parallel_styles attribute if it's None
        if hasattr(config, '_parallel_styles') and config._parallel_styles is None:
            config._parallel_styles = []
        if hasattr(config, 'parallel_styles') and config.parallel_styles is None:
            config.parallel_styles = []
        
        # Also check for rope_scaling related issues
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict) and 'type' in config.rope_scaling:
                # Ensure type is properly set
                if config.rope_scaling['type'] is None:
                    config.rope_scaling['type'] = 'default'
    
    return config


def test_qwen3_config_fix():
    """Test the Qwen3 configuration fix."""
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


def test_non_qwen3_config():
    """Test that the fix doesn't affect non-Qwen3 models."""
    print("\nTesting non-Qwen3 configuration...")
    
    class MockConfig:
        def __init__(self):
            self.model_type = 'llama'
            self._parallel_styles = None
            self.parallel_styles = None
    
    mock_config = MockConfig()
    original_parallel_styles = mock_config._parallel_styles
    
    # Apply the fix
    fixed_config = fix_qwen3_config(mock_config)
    
    # Should not change non-Qwen3 models
    assert fixed_config._parallel_styles == original_parallel_styles
    assert fixed_config.parallel_styles == original_parallel_styles
    
    print("✅ Non-Qwen3 configuration test passed!")


if __name__ == "__main__":
    print("Running simple Qwen3 fix tests...")
    
    # Test the configuration fix
    test_qwen3_config_fix()
    
    # Test that it doesn't affect other models
    test_non_qwen3_config()
    
    print("\n🎉 All tests passed! The fix function works correctly.") 