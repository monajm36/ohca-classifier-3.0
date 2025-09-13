"""
NLP OHCA Classifier v3.0 - Improved Methodology

A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases 
in medical discharge notes using improved methodology that addresses key concerns 
in medical AI development.

Key Improvements:
- Patient-level data splits to prevent data leakage
- Proper train/validation/test methodology  
- Optimal threshold finding and usage
- Enhanced clinical decision support
- Backward compatibility with legacy models
"""

__version__ = "3.0.0"
__author__ = "Mona Moukaddem"
__email__ = "mona.moukaddem@example.com"  # Update with actual email
__description__ = "BERT-based OHCA classifier with enhanced methodology"

# Version info
VERSION_INFO = {
    'major': 3,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

# Try to import main modules with error handling
try:
    # Training pipeline imports - v3.0 improved functions
    from .ohca_training_pipeline import (
        # Core v3.0 functions
        create_patient_level_splits,
        complete_improved_training_pipeline,
        complete_annotation_and_train_v3,
        find_optimal_threshold,
        evaluate_on_test_set,
        save_model_with_metadata,
        
        # Data preparation and training
        create_training_sample,
        prepare_training_data,
        train_ohca_model,
        evaluate_model,
        
        # Legacy functions (backward compatible)
        complete_training_pipeline,
        complete_annotation_and_train,
        create_training_sample_legacy,
        
        # Dataset class
        OHCATrainingDataset
    )
    
    TRAINING_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Training pipeline not fully available: {e}")
    TRAINING_AVAILABLE = False

try:
    # Inference imports - v3.0 enhanced functions
    from .ohca_inference import (
        # Core v3.0 functions (recommended)
        load_ohca_model_with_metadata,
        quick_inference_with_optimal_threshold,
        run_inference_with_optimal_threshold,
        process_large_dataset_with_optimal_threshold,
        analyze_predictions_enhanced,
        
        # Legacy functions (backward compatible)
        load_ohca_model,
        quick_inference,
        run_inference,
        process_large_dataset,
        test_model_on_sample,
        get_high_confidence_cases,
        analyze_predictions,
        
        # Dataset class
        OHCAInferenceDataset,
        
        # Utility functions
        categorize_confidence_with_optimal_threshold
    )
    
    INFERENCE_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Inference module not fully available: {e}")
    INFERENCE_AVAILABLE = False

# Define what gets imported with "from src import *"
__all__ = [
    # Version info
    '__version__', '__author__', '__description__',
    
    # Training functions (if available)
    'create_patient_level_splits',
    'complete_improved_training_pipeline', 
    'complete_annotation_and_train_v3',
    'find_optimal_threshold',
    'create_training_sample',
    'prepare_training_data',
    'train_ohca_model',
    'OHCATrainingDataset',
    
    # Inference functions (if available)
    'load_ohca_model_with_metadata',
    'quick_inference_with_optimal_threshold',
    'run_inference_with_optimal_threshold',
    'quick_inference',
    'load_ohca_model',
    'run_inference',
    'test_model_on_sample',
    'OHCAInferenceDataset',
    
    # Analysis functions
    'analyze_predictions_enhanced',
    'get_high_confidence_cases',
    
    # Batch processing
    'process_large_dataset_with_optimal_threshold',
    'process_large_dataset'
]

# Module status check
def check_module_status():
    """Check which modules are available"""
    status = {
        'training_available': TRAINING_AVAILABLE,
        'inference_available': INFERENCE_AVAILABLE,
        'version': __version__
    }
    return status

def get_version():
    """Get version information"""
    return f"OHCA Classifier v{__version__}"

def get_quick_start_info():
    """Print quick start information"""
    info = f"""
OHCA Classifier v{__version__} - Quick Start Guide

For pre-trained model usage:
    from src import quick_inference_with_optimal_threshold
    results = quick_inference_with_optimal_threshold(
        model_path="path/to/model", 
        data_path="your_data.csv"
    )

For training new models:
    from src import complete_improved_training_pipeline
    pipeline_result = complete_improved_training_pipeline(
        data_path="your_discharge_notes.csv"
    )

For more examples, see the examples/ folder.
Pre-trained model available at: https://huggingface.co/monajm36/ohca-classifier-v3
"""
    return info

# Initialize the module
def _initialize():
    """Initialize the module and print status"""
    status = check_module_status()
    
    print("="*60)
    print(f"NLP OHCA Classifier v{__version__} - Improved Methodology")
    print("="*60)
    print("Module Status:")
    print(f"  Training pipeline: {'✅ Available' if status['training_available'] else '❌ Not available'}")
    print(f"  Inference module: {'✅ Available' if status['inference_available'] else '❌ Not available'}")
    
    if status['training_available'] and status['inference_available']:
        print("  Status: 🟢 Fully operational")
        print("\nRecommended functions:")
        print("  • quick_inference_with_optimal_threshold() - Apply pre-trained model")
        print("  • complete_improved_training_pipeline() - Train new model with v3.0 methodology")
    elif status['inference_available']:
        print("  Status: 🟡 Inference only")
        print("  Note: Training functions not available")
    elif status['training_available']:
        print("  Status: 🟡 Training only") 
        print("  Note: Inference functions not available")
    else:
        print("  Status: 🔴 Limited functionality")
        print("  Note: Core modules not available")
    
    print("\nFor help: help(src) or src.get_quick_start_info()")
    print("Pre-trained model: https://huggingface.co/monajm36/ohca-classifier-v3")
    print("="*60)

# Run initialization when module is imported
_initialize()

# Cleanup
del _initialize
