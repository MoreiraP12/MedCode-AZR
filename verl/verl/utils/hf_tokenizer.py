import torch
from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


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


def hf_tokenizer(model_path_or_name: str,
                 trust_remote_code: bool = False,
                 use_fast: bool = True,
                 revision: str = None,
                 **kwargs) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Create a HuggingFace tokenizer with Qwen3 compatibility fixes.
    
    Args:
        model_path_or_name: Path or name of the model
        trust_remote_code: Whether to trust remote code
        use_fast: Whether to use fast tokenizer
        revision: Model revision
        **kwargs: Additional arguments
        
    Returns:
        Tokenizer instance
    """
    return AutoTokenizer.from_pretrained(
        model_path_or_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        revision=revision,
        **kwargs
    ) 