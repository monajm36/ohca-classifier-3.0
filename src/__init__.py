"""
NLP OHCA Classifier v3.0 - Improved Methodology
A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) 
cases in medical discharge notes using improved machine learning methodology.

Key Improvements in v3.0:
- Patient-level data splits to prevent data leakage
- Proper train/validation/test methodology  
- Optimal threshold finding and usage
- Larger annotation samples for better performance
- Unbiased evaluation framework

This package contains two main modules:
1. ohca_training_pipeline: Complete training pipeline with improved methodology
2. ohca_inference: Apply pre-trained models with optimal threshold support
"""

# Training pipeline imports - v3.0 with improvements
from .ohca_training_pipeline import (
    # Improved functions
    create_patient_level_splits,
    complete_improved_training_pipeline,
    complete_annotation_and_train_v3,
    find_optimal_threshold,
    evaluate_on_test_set,
    save_model_with_metadata,
    
    # Legacy functions (backward compatible)
    create_training_sample,
    prepare_training_data,
    train_ohca_model,
    evaluate_model,
    complete_training_pipeline,
    complete_annotation_and_train,
    
    # Dataset class
    OHCATrainingDataset
)

# Inference imports - v3.0 with optimal threshold support  
from .ohca_inference import (
    # New v3.0 functions with optimal threshold support
    load_ohca_model_with_metadata,
    run_inference_with_optimal_threshold,
    quick_inference_with_optimal_threshold,
    process_large_dataset_with_optimal_threshold,
    analyze_predictions_enhanced,
    
    # Legacy functions (backward compatible)
    load_ohca_model,
    run_inference,
    quick_inference,
    process_large_dataset,
    test_model_on_sample,
    get_high_confidence_cases,
    analyze_predictions,
    
    # Dataset class
    OHCAInferenceDataset
)

__version__ = "3.0.0"
__author__ = "Mona Moukaddem"
__email__ = "your.email@example.com"

# v3.0 improved functions (recommended)
__improved_training_functions__ = [
    "create_patient_level_splits",
    "complete_improved_training_pipeline", 
    "complete_annotation_and_train_v3",
    "find_optimal_threshold",
    "evaluate_on_test_set",
    "save_model_with_metadata"
]

__improved_inference_functions__ = [
    "load_ohca_model_with_metadata",
    "run_inference_with_optimal_threshold",
    "quick_inference_with_optimal_threshold", 
    "process_large_dataset_with_optimal_threshold",
    "analyze_predictions_enhanced"
]

# Legacy functions (maintained for backward compatibility)
__legacy_training_functions__ = [
    "create_training_sample",
    "prepare_training_data", 
    "train_ohca_model",
    "evaluate_model",
    "complete_training_pipeline",
    "complete_annotation_and_train",
    "OHCATrainingDataset"
]

__legacy_inference_functions__ = [
    "load_ohca_model",
    "run_inference",
    "quick_inference", 
    "process_large_dataset",
    "test_model_on_sample",
    "get_high_confidence_cases",
    "analyze_predictions",
    "OHCAInferenceDataset"
]

# All available functions
__all__ = (
    __improved_training_functions__ + 
    __improved_inference_functions__ + 
    __legacy_training_functions__ + 
    __legacy_inference_functions__
)

# Methodology information
__methodology_version__ = "3.0"
__improvements__ = [
    "Patient-level data splits prevent data leakage",
    "Proper train/validation/test methodology",
    "Optimal threshold finding and consistent usage", 
    "Larger annotation samples (800 train + 200 val)",
    "Unbiased evaluation on independent test set",
    "Enhanced clinical decision support",
    "Backward compatibility with legacy models"
]

def get_version_info():
    """Return detailed version and methodology information"""
    return {
        'version': __version__,
        'methodology_version': __methodology_version__,
        'improvements': __improvements__,
        'author': __author__,
        'recommended_functions': {
            'training': 'complete_improved_training_pipeline',
            'inference': 'quick_inference_with_optimal_threshold'
        }
    }

def print_welcome_message():
    """Print welcome message with key improvements"""
    print("="*60)
    print("NLP OHCA Classifier v3.0 - Improved Methodology")
    print("="*60)
    print("Key improvements addressing data scientist feedback:")
    for improvement in __improvements__:
        print(f"✅ {improvement}")
    print()
    print("Recommended functions:")
    print("• Training: complete_improved_training_pipeline()")
    print("• Inference: quick_inference_with_optimal_threshold()")
    print()
    print("Legacy functions maintained for backward compatibility.")
    print("="*60)

# Print welcome message when package is imported
print_welcome_message()
