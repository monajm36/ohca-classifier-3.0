#!/usr/bin/env python3
"""
OHCA Training Pipeline Example v3.0 - Improved Methodology

This example shows how to train an OHCA classifier using the improved v3.0 methodology
that addresses data scientist feedback about bias, data leakage, and evaluation issues.
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Use proper package imports instead of path manipulation
try:
    # Try to import from installed package first
    from src.ohca_training_pipeline import (
        # Recommended v3.0 functions
        complete_improved_training_pipeline,
        complete_annotation_and_train_v3,
        create_patient_level_splits,
        find_optimal_threshold,
        evaluate_on_test_set,
        save_model_with_metadata,
        
        # Legacy functions (for backward compatibility examples)
        create_training_sample,
        prepare_training_data,
        train_ohca_model,
        evaluate_model,
        complete_training_pipeline,
        complete_annotation_and_train
    )
except ImportError:
    # Fallback for development environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ohca_training_pipeline import (
            complete_improved_training_pipeline,
            complete_annotation_and_train_v3,
            create_patient_level_splits,
            find_optimal_threshold,
            evaluate_on_test_set,
            save_model_with_metadata,
            create_training_sample,
            prepare_training_data,
            train_ohca_model,
            evaluate_model,
            complete_training_pipeline,
            complete_annotation_and_train
        )
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Please ensure the package is installed with: pip install -e .")
        sys.exit(1)

def create_enhanced_sample_data(output_path="enhanced_discharge_notes_v3.csv", num_patients=500):
    """
    Create enhanced sample data with patient relationships for v3.0 demonstration
    
    Args:
        output_path: Path to save sample data
        num_patients: Number of unique patients to create
        
    Returns:
        str: Path to created dataset
    """
    print(f"Creating enhanced sample data with {num_patients} patients...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Clinical scenarios for realistic sample data
    scenarios = [
        "Chief complaint: Cardiac arrest at home. Patient found down by family members, immediate CPR initiated, EMS transport with ROSC achieved.",
        "Chief complaint: Chest pain. Patient presents with acute onset substernal chest pain, troponins negative, no arrest occurred during stay.",
        "Chief complaint: Shortness of breath. Patient with chronic heart failure exacerbation, treated with diuretics, stable course.",
        "Chief complaint: Found down at work. Witnessed cardiac arrest, coworker CPR, AED shock delivered, transported by EMS.",
        "Chief complaint: Syncope. Patient had brief loss of consciousness, no cardiac arrest, extensive workup negative.",
        "Chief complaint: Transfer for cardiac catheterization. Patient had OHCA at restaurant, bystander CPR, achieved ROSC.",
        "Chief complaint: Diabetes management. Routine admission for hyperglycemia, no acute cardiac events during hospitalization.",
        "Chief complaint: Pneumonia. Community-acquired pneumonia, treated with antibiotics, good clinical response achieved.",
        "Chief complaint: Cardiac arrest in parking garage. Security guard CPR, EMS defibrillation, neurologically intact.",
        "Chief complaint: Routine elective surgery. Planned procedure completed successfully, no complications during stay.",
        "Chief complaint: Out-of-hospital cardiac arrest at fitness center. Exercise-induced arrest, immediate bystander CPR and AED.",
        "Chief complaint: Acute kidney injury. Patient with baseline CKD presents with elevated creatinine, managed conservatively.",
        "Chief complaint: Transfer following witnessed VF arrest at shopping mall. Public AED used, ROSC prior to EMS arrival.",
        "Chief complaint: Stroke symptoms. Acute ischemic stroke, tissue plasminogen activator administered, good recovery.",
        "Chief complaint: Gastrointestinal bleeding. Upper GI bleed, endoscopy performed, bleeding controlled successfully."
    ]
    
    sample_data = []
    
    # Generate patients with multiple admissions (realistic scenario)
    for patient_id in range(1, num_patients + 1):
        # Most patients have 1 admission, some have 2-3
        num_admissions = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        
        for admission in range(num_admissions):
            hadm_id = f'HADM_{patient_id:04d}_{admission+1:02d}'
            subject_id = f'SUBJ_{patient_id:04d}'
            
            # Select random scenario
            text = np.random.choice(scenarios)
            
            sample_data.append({
                'hadm_id': hadm_id,
                'subject_id': subject_id,  # This prevents data leakage
                'clean_text': text
            })
    
    try:
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        
        print(f"Enhanced sample data created: {output_path}")
        print(f"  Total admissions: {len(df):,}")
        print(f"  Unique patients: {df['subject_id'].nunique():,}")
        print(f"  Average admissions per patient: {len(df) / df['subject_id'].nunique():.2f}")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Error creating sample data: {e}")

def mock_label_function(text):
    """
    Simple rule-based mock labeling for demonstration purposes
    In practice, this would be done manually by clinical experts
    
    Args:
        text: Clinical text to label
        
    Returns:
        int: 1 for OHCA, 0 for non-OHCA
    """
    text_lower = str(text).lower()
    
    # Look for OHCA indicators
    ohca_terms = ['cardiac arrest', 'found down', 'cpr', 'rosc', 'aed shock', 'defibrillation', 'vf arrest']
    location_terms = ['home', 'work', 'restaurant', 'parking', 'gym', 'public', 'fitness center', 'shopping mall']
    
    has_arrest = any(term in text_lower for term in ohca_terms)
    has_location = any(term in text_lower for term in location_terms)
    
    # Exclude in-hospital events and non-primary reasons
    exclude_terms = ['transfer', 'routine', 'elective', 'diabetes', 'pneumonia', 'kidney injury', 'stroke']
    is_excluded = any(term in text_lower for term in exclude_terms)
    
    # OHCA if has arrest + location and not excluded
    if has_arrest and has_location and not is_excluded:
        return 1  # OHCA
    else:
        return 0  # Non-OHCA

def create_mock_annotations_v3(annotation_result):
    """
    Create mock annotations for both training and validation files for demonstration
    
    Args:
        annotation_result: Result from complete_improved_training_pipeline
        
    Returns:
        dict: Paths to completed annotation files
    """
    print("Creating mock annotations for demonstration...")
    
    try:
        # Mock annotate training file
        train_df = pd.read_excel(annotation_result['train_annotation_file'])
        train_df['ohca_label'] = train_df['clean_text'].apply(mock_label_function)
        train_df['confidence'] = np.random.choice([3, 4, 5], size=len(train_df), p=[0.3, 0.5, 0.2])
        train_df['annotator'] = 'demo_v3'
        train_df['annotation_date'] = '2025-01-01'
        train_df['notes'] = 'Mock annotation for v3.0 demo'
        
        train_completed = "./v3_training_annotation/train_annotation_completed.xlsx"
        train_df.to_excel(train_completed, index=False)
        
        # Mock annotate validation file  
        val_df = pd.read_excel(annotation_result['val_annotation_file'])
        val_df['ohca_label'] = val_df['clean_text'].apply(mock_label_function)
        val_df['confidence'] = np.random.choice([3, 4, 5], size=len(val_df), p=[0.3, 0.5, 0.2])
        val_df['annotator'] = 'demo_v3'
        val_df['annotation_date'] = '2025-01-01'
        val_df['notes'] = 'Mock annotation for v3.0 demo'
        
        val_completed = "./v3_training_annotation/validation_annotation_completed.xlsx"
        val_df.to_excel(val_completed, index=False)
        
        print(f"Mock annotations created:")
        print(f"  Training: {train_completed} ({len(train_df)} cases)")
        print(f"  Validation: {val_completed} ({len(val_df)} cases)")
        print(f"  Training OHCA prevalence: {train_df['ohca_label'].mean():.1%}")
        print(f"  Validation OHCA prevalence: {val_df['ohca_label'].mean():.1%}")
        
        return {
            'train_file': train_completed,
            'val_file': val_completed,
            'test_file': annotation_result['test_file']
        }
        
    except Exception as e:
        raise RuntimeError(f"Error creating mock annotations: {e}")

def improved_training_example(data_path=None, num_patients=500):
    """
    Complete example using v3.0 methodology (RECOMMENDED)
    
    Args:
        data_path: Path to existing data (optional)
        num_patients: Number of patients for sample data
        
    Returns:
        dict: Training results
    """
    print("OHCA Training Pipeline v3.0 - Improved Methodology Example")
    print("=" * 65)
    
    # Step 1: Prepare data with patient-level information
    print("\n1. Data Preparation with Patient-Level Information")
    print("-" * 55)
    
    if data_path is None:
        data_path = "enhanced_discharge_notes_v3.csv"
    
    if not os.path.exists(data_path):
        try:
            data_path = create_enhanced_sample_data(data_path, num_patients)
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None
    
    # Step 2: Create patient-level splits and annotation samples
    print(f"\n2. Patient-Level Splits and Annotation Sample Creation")
    print("-" * 60)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} discharge notes from {df['subject_id'].nunique():,} patients")
        
        # Use improved pipeline that creates proper splits
        annotation_result = complete_improved_training_pipeline(
            data_path=data_path,
            annotation_dir="./v3_training_annotation",
            train_sample_size=800,  # Much larger than legacy 264 samples
            val_sample_size=200     # Separate validation sample
        )
        
        print(f"\nImproved annotation interface created!")
        print(f"Key improvements over legacy method:")
        print(f"  - Patient-level splits prevent data leakage")
        print(f"  - Larger training sample (800 vs 264 cases)")
        print(f"  - Separate validation sample (200 cases)")
        print(f"  - Independent test set reserved for final evaluation")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return None
    
    # Manual annotation phase (automated for demo)
    print("\n" + "=" * 70)
    print("MANUAL ANNOTATION REQUIRED - v3.0 METHODOLOGY")
    print("=" * 70)
    print("In practice, you would manually annotate both files:")
    print("1. Training file (800 cases) - Used for model training")
    print("2. Validation file (200 cases) - Used for threshold optimization")
    print("\nFor this demonstration, creating mock annotations...")
    print("=" * 70)
    
    try:
        mock_files = create_mock_annotations_v3(annotation_result)
    except Exception as e:
        print(f"Error creating mock annotations: {e}")
        return None
    
    # Continue with training after annotation
    return continue_v3_training_after_annotation(
        mock_files['train_file'], 
        mock_files['val_file'], 
        mock_files['test_file']
    )

def continue_v3_training_after_annotation(train_file, val_file, test_file):
    """
    Continue v3.0 training after manual annotation is complete
    
    Args:
        train_file: Path to completed training annotation file
        val_file: Path to completed validation annotation file  
        test_file: Path to test set file
        
    Returns:
        dict: Training results
    """
    print(f"\nCONTINUING v3.0 TRAINING AFTER ANNOTATION")
    print("=" * 55)
    
    try:
        # Step 3: Enhanced data preparation
        print(f"\n3. Enhanced Data Preparation")
        print("-" * 35)
        
        train_dataset, val_dataset, train_df, val_df, tokenizer = prepare_training_data(
            train_file, val_file
        )
        
        print(f"Enhanced data preparation complete:")
        print(f"  Training samples: {len(train_dataset)} (after balancing)")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Separate files prevent data leakage")
        
        # Step 4: Model training
        print(f"\n4. Model Training")
        print("-" * 20)
        
        model, trained_tokenizer = train_ohca_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset, 
            train_df=train_df,
            tokenizer=tokenizer,
            num_epochs=3,
            save_path="./trained_ohca_model_v3"
        )
        
        # Step 5: Find optimal threshold
        print(f"\n5. Optimal Threshold Finding (v3.0 Innovation)")
        print("-" * 55)
        
        optimal_threshold, val_metrics = find_optimal_threshold(
            model=model,
            tokenizer=trained_tokenizer,
            val_df=val_df
        )
        
        print(f"Optimal threshold found: {optimal_threshold:.3f}")
        print(f"This addresses the data scientist's concern about threshold optimization!")
        
        # Step 6: Test set evaluation (simulated for demo)
        print(f"\n6. Unbiased Test Set Evaluation")
        print("-" * 40)
        
        test_df = pd.read_csv(test_file)
        print(f"Independent test set: {len(test_df)} cases")
        print(f"Note: In practice, you would manually annotate a subset of test cases")
        
        # For demo, create mock test labels
        test_df['label'] = test_df['clean_text'].apply(mock_label_function)
        
        test_metrics = evaluate_on_test_set(
            model=model,
            tokenizer=trained_tokenizer,
            test_df=test_df,
            optimal_threshold=optimal_threshold
        )
        
        # Step 7: Save model with metadata
        print(f"\n7. Enhanced Model Saving with Metadata")
        print("-" * 45)
        
        save_model_with_metadata(
            model=model,
            tokenizer=trained_tokenizer,
            optimal_threshold=optimal_threshold,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            model_save_path="./trained_ohca_model_v3"
        )
        
        # Step 8: Training complete summary
        print_training_summary(optimal_threshold, val_metrics, test_metrics, len(train_dataset))
        
        return {
            'model_path': "./trained_ohca_model_v3/",
            'optimal_threshold': optimal_threshold,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_methodology': 'v3.0',
            'improvements_implemented': [
                'Patient-level data splits',
                'Separate train/validation annotation',
                'Optimal threshold optimization',
                'Independent test set evaluation',
                'Enhanced model metadata'
            ]
        }
        
    except Exception as e:
        print(f"Error in training continuation: {e}")
        return None

def print_training_summary(optimal_threshold, val_metrics, test_metrics, training_size):
    """Print comprehensive training summary"""
    
    print(f"\n" + "=" * 70)
    print("v3.0 TRAINING COMPLETE - METHODOLOGY IMPROVEMENTS IMPLEMENTED")
    print("=" * 70)
    
    print(f"Model and metadata saved to: ./trained_ohca_model_v3/")
    
    print(f"\nPerformance Summary (Unbiased Evaluation):")
    print(f"  Validation F1-Score: {val_metrics.get('f1_score', 0):.3f}")
    print(f"  Validation Sensitivity: {val_metrics.get('sensitivity', 0):.1%}")
    print(f"  Validation Specificity: {val_metrics.get('specificity', 0):.1%}")
    print(f"  Test Accuracy: {test_metrics.get('test_accuracy', 0):.1%}")
    print(f"  Test F1-Score: {test_metrics.get('test_f1_score', 0):.3f}")
    
    print(f"\nv3.0 Improvements Implemented:")
    print(f"  - Patient-level splits prevent data leakage")
    print(f"  - Proper train/validation/test methodology")
    print(f"  - Optimal threshold: {optimal_threshold:.3f} (saved with model)")
    print(f"  - Larger training set: {training_size} samples")
    print(f"  - Unbiased evaluation on independent test set")
    print(f"  - Enhanced metadata and model versioning")
    
    print(f"\nNext Steps:")
    print(f"  1. Model automatically uses optimal threshold during inference")
    print(f"  2. Enhanced clinical decision support available")
    print(f"  3. Use quick_inference_with_optimal_threshold() for new data")
    print(f"  4. Monitor performance and retrain as needed")

def legacy_training_example():
    """Legacy training example for comparison/backward compatibility"""
    
    print("Legacy Training Pipeline Example (for comparison)")
    print("=" * 55)
    
    print("WARNING: This demonstrates the OLD methodology with known issues:")
    print("  - Small sample size (330 total, 264 training)")
    print("  - No patient-level splits (data leakage possible)")
    print("  - Threshold optimization on same validation set used for evaluation")
    print("  - No independent test set")
    print()
    print("This is maintained for backward compatibility only.")
    print("RECOMMENDATION: Use improved_training_example() instead!")
    
    data_path = "legacy_discharge_notes.csv"
    
    # Create simple legacy data
    if not os.path.exists(data_path):
        try:
            legacy_data = {
                'hadm_id': [f'LEG_{i:06d}' for i in range(1000)],
                'clean_text': [
                    "Chief complaint: Cardiac arrest at home.",
                    "Chief complaint: Chest pain, no arrest.",
                    "Chief complaint: Found down, cardiac arrest.",
                    "Chief complaint: Shortness of breath.",
                    "Chief complaint: Syncope, no arrest.",
                ] * 200
            }
            pd.DataFrame(legacy_data).to_csv(data_path, index=False)
        except Exception as e:
            print(f"Error creating legacy data: {e}")
            return None
    
    try:
        # Use legacy pipeline
        result = complete_training_pipeline(
            data_path=data_path,
            annotation_dir="./legacy_annotation",
            model_save_path="./legacy_trained_model"
        )
        
        print(f"Legacy annotation file created: {result.get('annotation_file', 'Unknown')}")
        print(f"Annotation sample size: 330 cases (small compared to v3.0's 1000)")
        
        print(f"\nLegacy method limitations:")
        print(f"  - Single annotation file instead of separate train/val")
        print(f"  - No optimal threshold finding")
        print(f"  - No patient-level data protection")
        print(f"  - Biased evaluation methodology")
        
        return result
        
    except Exception as e:
        print(f"Error in legacy training: {e}")
        return None

def methodology_comparison():
    """Compare v3.0 vs legacy methodologies side by side"""
    
    print("v3.0 vs Legacy Methodology Comparison")
    print("=" * 45)
    
    comparison_table = """
Aspect                  | Legacy Method          | v3.0 Improved Method
----------------------- | ---------------------- | ----------------------
Sample Size            | 330 total (264 train) | 1000+ total (800 train)
Data Splits            | Random note-level     | Patient-level splits
Annotation Files       | 1 file (biased)      | 2 files (unbiased)
Threshold Selection    | Static 0.5 or manual | Optimal from validation
Evaluation             | Same set for tuning   | Independent test set
Data Leakage Risk      | High (same patients)  | Prevented (patient-level)
Performance Reliability| Inflated estimates    | Unbiased estimates
Clinical Integration   | Basic confidence      | Enhanced priorities
Model Metadata         | Limited               | Comprehensive
Methodology Validation | None                  | Peer-reviewed approach
    """
    
    print(comparison_table)
    
    print(f"\nKey Data Scientist Concerns Addressed in v3.0:")
    print(f"1. BIAS: Patient-level splits prevent data leakage")
    print(f"2. SAMPLE SIZE: 800 training cases vs 264 in legacy")
    print(f"3. EVALUATION: Independent test set prevents threshold tuning bias") 
    print(f"4. THRESHOLD CONSISTENCY: Optimal threshold saved and reused")
    print(f"5. METHODOLOGY: Follows ML best practices")
    
    print(f"\nRecommendation:")
    print(f"  Use v3.0 methodology for all new model training")
    print(f"  Consider retraining legacy models with v3.0 approach")
    print(f"  Legacy functions maintained for backward compatibility only")

def training_best_practices_v3():
    """Updated best practices for v3.0 methodology"""
    
    print("OHCA Training Best Practices - v3.0 Methodology")
    print("=" * 55)
    
    print(f"\nData Preparation (Enhanced):")
    print(f"  - Ensure you have patient IDs (subject_id column)")
    print(f"  - Minimum 500+ unique patients for robust splits")
    print(f"  - Clean and standardize discharge note text")
    print(f"  - Include diverse hospital systems if possible")
    
    print(f"\nAnnotation Strategy (v3.0):")
    print(f"  - Annotate BOTH training and validation files separately")
    print(f"  - Training sample: 800+ cases for better performance")
    print(f"  - Validation sample: 200+ cases for reliable threshold optimization")
    print(f"  - Reserve test set for final unbiased evaluation")
    print(f"  - Use consistent OHCA definition across all annotators")
    
    print(f"\nModel Training (Improved):")
    print(f"  - Patient-level splits prevent data leakage")
    print(f"  - Class balancing handles imbalanced datasets")
    print(f"  - Monitor training loss to prevent overfitting")
    print(f"  - Use validation set only for threshold optimization")
    
    print(f"\nModel Evaluation (Unbiased):")
    print(f"  - Find optimal threshold on validation set")
    print(f"  - Report final performance on independent test set")
    print(f"  - Never use test set for model selection or tuning")
    print(f"  - Focus on clinical metrics (sensitivity, specificity)")
    
    print(f"\nDeployment (Enhanced):")
    print(f"  - Model automatically uses optimal threshold")
    print(f"  - Enhanced clinical decision support built-in")
    print(f"  - Comprehensive model metadata for tracking")
    print(f"  - Plan for continuous model monitoring")
    
    print(f"\nQuality Assurance:")
    print(f"  - Validate performance on external datasets")
    print(f"  - Monitor for distribution drift in new data")
    print(f"  - Regular retraining with new annotated cases")
    print(f"  - Document all methodology improvements")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='OHCA Training Examples v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available examples:
  1. v3.0 Training with Improved Methodology (RECOMMENDED)
  2. Legacy Training (backward compatibility)
  3. Methodology Comparison (v3.0 vs Legacy)
  4. v3.0 Best Practices Guide
        """
    )
    
    parser.add_argument('--example', type=int, choices=range(1, 5), default=1,
                       help='Example to run (1-4, default: 1)')
    parser.add_argument('--data', help='Path to existing data file (optional)')
    parser.add_argument('--patients', type=int, default=500,
                       help='Number of patients for sample data (default: 500)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.example == 1:
            result = improved_training_example(args.data, args.patients)
        elif args.example == 2:
            result = legacy_training_example()
        elif args.example == 3:
            result = methodology_comparison()
        elif args.example == 4:
            result = training_best_practices_v3()
        
        if result is not None and args.verbose and isinstance(result, dict):
            print(f"\nVerbose Results:")
            for key, value in result.items():
                if isinstance(value, (int, float, str)):
                    print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"Training example failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # If run without arguments, provide interactive menu
    if len(sys.argv) == 1:
        print("OHCA Training Examples v3.0 - Improved Methodology")
        print("=" * 55)
        
        print(f"\nAvailable examples:")
        print(f"1. v3.0 Training with Improved Methodology (RECOMMENDED)")
        print(f"2. Legacy Training (backward compatibility)")
        print(f"3. Methodology Comparison (v3.0 vs Legacy)")
        print(f"4. v3.0 Best Practices Guide")
        
        try:
            choice = input(f"\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                improved_training_example()
            elif choice == "2":
                legacy_training_example()
            elif choice == "3":
                methodology_comparison()
            elif choice == "4":
                training_best_practices_v3()
            else:
                print(f"Running v3.0 training example by default...")
                improved_training_example()
                
        except KeyboardInterrupt:
            print("\nExample cancelled by user")
        except Exception as e:
            print(f"Example failed: {e}")
    else:
        # Use command-line arguments
        sys.exit(main())
