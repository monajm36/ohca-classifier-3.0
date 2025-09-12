cat > src/__init__.py << 'EOF'
"""
NLP OHCA Classifier v3.0 - Improved Methodology
"""

# Training pipeline imports - only working functions
from .ohca_training_pipeline import (
    # v3.0 improved functions
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
    complete_training_pipeline,
    complete_annotation_and_train,
    
    # Dataset class
    OHCATrainingDataset
)

# Inference imports
from .ohca_inference import (
    load_ohca_model,
    run_inference,
    quick_inference,
    process_large_dataset,
    test_model_on_sample,
    get_high_confidence_cases,
    analyze_predictions,
    OHCAInferenceDataset
)

__version__ = "3.0.0"
__author__ = "Mona Moukaddem"

print("NLP OHCA Classifier v3.0 loaded successfully!")
EOF
