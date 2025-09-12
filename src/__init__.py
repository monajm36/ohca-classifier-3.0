"""
NLP OHCA Classifier v3.0
A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases
"""

__version__ = "3.0.0"
__author__ = "Mona Moukaddem"

# Only import functions that actually exist
try:
    from .ohca_training_pipeline import (
        create_training_sample,
        complete_training_pipeline,
        complete_annotation_and_train
    )
    
    from .ohca_inference import (
        load_ohca_model,
        quick_inference,
        run_inference
    )
    
    print("NLP OHCA Classifier v3.0 loaded successfully")
    
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some functions may not be available")

__all__ = [
    "create_training_sample",
    "complete_training_pipeline", 
    "complete_annotation_and_train",
    "load_ohca_model",
    "quick_inference",
    "run_inference"
]
