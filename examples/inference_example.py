#!/usr/bin/env python3
"""
OHCA Inference Example v3.0 - Enhanced with Optimal Threshold Support

This example shows how to use pre-trained OHCA classifiers with the improved
v3.0 methodology, including optimal threshold usage and enhanced clinical 
decision support.
"""

import pandas as pd
import sys
import os
import argparse
from pathlib import Path

# Use proper package imports instead of path manipulation
try:
    # Try to import from installed package first
    from src.ohca_inference import (
        # Recommended v3.0 functions
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
        analyze_predictions
    )
except ImportError:
    # Fallback for development environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ohca_inference import (
            load_ohca_model_with_metadata,
            run_inference_with_optimal_threshold,
            quick_inference_with_optimal_threshold,
            process_large_dataset_with_optimal_threshold,
            analyze_predictions_enhanced,
            load_ohca_model,
            run_inference,
            quick_inference,
            process_large_dataset,
            test_model_on_sample,
            get_high_confidence_cases,
            analyze_predictions
        )
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Please ensure the package is installed with: pip install -e .")
        sys.exit(1)

def create_sample_data(output_path="sample_new_data_v3.csv", size=20):
    """
    Create sample data for inference demonstration
    
    Args:
        output_path: Path to save sample data
        size: Number of sample records
        
    Returns:
        str: Path to created sample data
    """
    print(f"Creating sample data for v3.0 demonstration...")
    
    # Enhanced sample cases with realistic clinical scenarios
    sample_cases = [
        "Chief complaint: Cardiac arrest at home. Family initiated CPR immediately, EMS transported to hospital with ROSC achieved.",
        "Chief complaint: Chest pain. Patient has stable angina, no cardiac arrest occurred during admission, negative troponins.",
        "Chief complaint: Found down at work. Witnessed cardiac arrest, coworker performed CPR, AED used with successful ROSC.",
        "Chief complaint: Shortness of breath. CHF exacerbation, treated with diuretics, stable clinical course throughout.",
        "Chief complaint: Syncope. Brief loss of consciousness, no arrest occurred, negative cardiac workup completed.",
        "Chief complaint: Transfer for cardiac catheterization. OHCA at restaurant, bystander CPR given, neurologically intact.",
        "Chief complaint: Diabetes management. Routine admission for glucose control, no acute events during stay.",
        "Chief complaint: Cardiac arrest in parking garage. CPR by security guard, EMS achieved ROSC after 15 minutes.",
        "Chief complaint: Pneumonia. Community-acquired pneumonia, treated with antibiotics, good clinical response.",
        "Chief complaint: Collapse at gym. Witnessed VF arrest, immediate bystander CPR and defibrillation provided.",
        "Chief complaint: Abdominal pain. Acute appendicitis, underwent successful appendectomy, routine recovery.",
        "Chief complaint: Found unresponsive at home. Cardiac arrest witnessed by spouse, immediate CPR initiated.",
        "Chief complaint: Hypertensive emergency. Severe HTN, treated with IV medications, no cardiac complications.",
        "Chief complaint: Cardiac arrest at shopping mall. Bystander CPR, public AED used, ROSC prior to EMS.",
        "Chief complaint: Elective surgery. Planned procedure completed successfully, no intraoperative complications.",
        "Chief complaint: Out-of-hospital arrest. Found down in driveway, neighbor CPR, transported with ROSC.",
        "Chief complaint: Migraine headache. Severe headache, treated with medications, neurologic exam normal.",
        "Chief complaint: Cardiac arrest during exercise. Collapsed while jogging, immediate CPR by witnesses.",
        "Chief complaint: Upper respiratory infection. Viral syndrome, treated symptomatically, improved clinically.",
        "Chief complaint: Witnessed collapse with loss of consciousness. Cardiac arrest, bystander CPR given immediately."
    ]
    
    # Create balanced sample dataset
    sample_data = {
        'hadm_id': [f'V3_{i:06d}' for i in range(1, size + 1)],
        'clean_text': (sample_cases * (size // len(sample_cases) + 1))[:size]
    }
    
    try:
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"Sample data created: {output_path} ({len(df)} records)")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error creating sample data: {e}")

def check_model_version(model_path):
    """
    Check if model is v3.0 (with metadata) or legacy
    
    Args:
        model_path: Path to model directory
        
    Returns:
        str: 'v3.0', 'legacy', or 'not_found'
    """
    if not os.path.exists(model_path):
        return 'not_found'
    
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_path):
        return 'v3.0'
    else:
        return 'legacy'

def improved_inference_example(model_path="./trained_ohca_model_v3", 
                             data_path="sample_new_data_v3.csv"):
    """
    Example using v3.0 methodology with optimal threshold (RECOMMENDED)
    
    Args:
        model_path: Path to v3.0 model
        data_path: Path to data for inference
        
    Returns:
        DataFrame: Inference results
    """
    print("OHCA Inference v3.0 - Improved Methodology Example")
    print("=" * 60)
    
    # Check model availability
    model_status = check_model_version(model_path)
    
    if model_status == 'not_found':
        print(f"Model not found at: {model_path}")
        print("Please train a model using complete_improved_training_pipeline() first.")
        return None
    elif model_status == 'legacy':
        print("Model found but no metadata detected. This appears to be a legacy model.")
        print("Consider retraining with v3.0 methodology for optimal performance.")
        return legacy_inference_example(model_path, data_path)
    
    # Ensure sample data exists
    if not os.path.exists(data_path):
        try:
            create_sample_data(data_path)
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None
    
    # Step 1: Load v3.0 model with metadata
    print(f"\nSTEP 1: Loading v3.0 Model with Metadata")
    print("-" * 50)
    
    try:
        model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
        
        print(f"Model loaded successfully!")
        print(f"   Model version: {metadata.get('model_version', 'unknown')}")
        print(f"   Optimal threshold: {optimal_threshold:.3f}")
        print(f"   Training date: {metadata.get('training_date', 'unknown')}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Step 2: Load and validate data
    print(f"\nSTEP 2: Loading Data for Inference")
    print("-" * 40)
    
    try:
        new_df = pd.read_csv(data_path)
        print(f"Loaded {len(new_df)} cases for inference")
        
        # Validate data format
        required_cols = ['hadm_id', 'clean_text']
        missing_cols = [col for col in required_cols if col not in new_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Step 3: Run inference with optimal threshold
    print(f"\nSTEP 3: Running Inference with Optimal Threshold")
    print("-" * 55)
    
    try:
        results = run_inference_with_optimal_threshold(
            model=model,
            tokenizer=tokenizer,
            inference_df=new_df,
            optimal_threshold=optimal_threshold,
            batch_size=8,
            output_path="./v3_inference_results.csv"
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    # Step 4: Analyze enhanced results
    print(f"\nSTEP 4: Enhanced Results Analysis")
    print("-" * 45)
    
    try:
        analysis = analyze_predictions_enhanced(results)
        
        # Show clinical priorities
        if 'clinical_priority' in results.columns:
            print(f"\nClinical Priority Cases:")
            print("-" * 30)
            
            immediate = results[results['clinical_priority'] == 'Immediate Review']
            priority = results[results['clinical_priority'] == 'Priority Review'] 
            clinical = results[results['clinical_priority'] == 'Clinical Review']
            
            if len(immediate) > 0:
                print(f"Immediate Review ({len(immediate)} cases):")
                for _, row in immediate.iterrows():
                    hadm_id = row['hadm_id']
                    prob = row['ohca_probability']
                    text = new_df[new_df['hadm_id'] == hadm_id]['clean_text'].iloc[0]
                    print(f"   {hadm_id}: p={prob:.3f} - {text[:80]}...")
            
            if len(priority) > 0:
                print(f"\nPriority Review ({len(priority)} cases):")
                for _, row in priority.head(3).iterrows():  # Show first 3
                    hadm_id = row['hadm_id']
                    prob = row['ohca_probability']  # Fixed typo from original
                    print(f"   {hadm_id}: p={prob:.3f}")
        
    except Exception as e:
        print(f"Warning: Error in analysis: {e}")
    
    # Step 5: Compare with legacy thresholds
    print(f"\nSTEP 5: Threshold Comparison")
    print("-" * 35)
    
    optimal_detections = results['ohca_prediction'].sum()
    static_050_detections = results.get('prediction_050', pd.Series()).sum()
    static_070_detections = results.get('prediction_070', pd.Series()).sum()
    
    print(f"Optimal threshold ({optimal_threshold:.3f}): {optimal_detections} OHCA cases")
    print(f"Static threshold (0.5): {static_050_detections} OHCA cases")
    print(f"Static threshold (0.7): {static_070_detections} OHCA cases")
    
    if optimal_detections != static_050_detections:
        print(f"Optimal threshold shows different results than static 0.5!")
        print(f"This demonstrates the value of threshold optimization.")
    
    # Step 6: Clinical workflow integration
    print(f"\nSTEP 6: Clinical Workflow Integration")
    print("-" * 45)
    
    print("Recommended workflow based on v3.0 results:")
    print("1. Immediate Review cases → Priority manual review")
    print("2. Priority Review cases → Clinical team review")  
    print("3. Clinical Review cases → Consider for quality checks")
    print("4. Lower priority cases → Routine processing")
    
    # Show expected clinical impact
    total_cases = len(results)
    if 'clinical_priority' in results.columns:
        high_priority_cases = len(results[
            results['clinical_priority'].isin(['Immediate Review', 'Priority Review'])
        ])
        
        if high_priority_cases > 0:
            efficiency_gain = (total_cases - high_priority_cases) / total_cases * 100
            print(f"\nExpected Efficiency Gains:")
            print(f"   Focus review on {high_priority_cases}/{total_cases} cases ({high_priority_cases/total_cases*100:.1f}%)")
            print(f"   Potential {efficiency_gain:.1f}% reduction in manual review burden")
    
    print(f"\nv3.0 INFERENCE COMPLETE!")
    print("=" * 50)
    print("Key v3.0 advantages demonstrated:")
    print("- Optimal threshold from validation set")
    print("- Enhanced clinical decision support")
    print("- Improved confidence categorization")
    print("- Better workflow integration")
    
    return results

def quick_inference_v3_example(model_path="./trained_ohca_model_v3", 
                              data_path="sample_new_data_v3.csv"):
    """
    Quick inference using v3.0 convenience function (RECOMMENDED)
    
    Args:
        model_path: Path to model
        data_path: Path to data
        
    Returns:
        DataFrame: Inference results
    """
    print("Quick v3.0 Inference Example")
    print("=" * 35)
    
    # Ensure data exists
    if not os.path.exists(data_path):
        try:
            create_sample_data(data_path)
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None
    
    # Check model version
    model_status = check_model_version(model_path)
    
    try:
        if model_status == 'v3.0':
            print("Detected v3.0 model - using optimal threshold")
            
            results = quick_inference_with_optimal_threshold(
                model_path=model_path,
                data_path=data_path,
                output_path="./quick_v3_results.csv"
            )
            
            print(f"\nv3.0 Quick Results:")
            if 'optimal_threshold_used' in results.columns:
                print(f"   Optimal threshold used: {results['optimal_threshold_used'].iloc[0]:.3f}")
            print(f"   OHCA detected: {results['ohca_prediction'].sum()}")
            if 'clinical_priority' in results.columns:
                immediate_count = (results['clinical_priority'] == 'Immediate Review').sum()
                print(f"   Immediate review needed: {immediate_count}")
            
        else:
            print("No v3.0 model found - falling back to legacy method")
            results = quick_inference(
                model_path=model_path,
                data_path=data_path,
                output_path="./quick_legacy_results.csv"
            )
        
        return results
        
    except Exception as e:
        print(f"Error in quick inference: {e}")
        return None

def legacy_inference_example(model_path="./trained_ohca_model", 
                           data_path="sample_legacy_data.csv"):
    """
    Legacy inference example for backward compatibility
    
    Args:
        model_path: Path to legacy model
        data_path: Path to data
        
    Returns:
        DataFrame: Inference results
    """
    print("Legacy Inference Example (Backward Compatibility)")
    print("=" * 55)
    
    if not os.path.exists(model_path):
        print(f"Legacy model not found at: {model_path}")
        print("Please train a model first or use the v3.0 methodology.")
        return None
    
    print("Using legacy inference method with static threshold 0.5")
    
    # Create simple sample data if needed
    if not os.path.exists(data_path):
        sample_data = {
            'hadm_id': [f'LEG_{i:03d}' for i in range(1, 11)],
            'clean_text': [
                "Chief complaint: Cardiac arrest at home.",
                "Chief complaint: Chest pain, no arrest.",
                "Chief complaint: Found down, cardiac arrest.",
                "Chief complaint: Shortness of breath.",
                "Chief complaint: Syncope, no arrest.",
                "Chief complaint: Transfer for cardiac arrest.",
                "Chief complaint: Diabetes management.",
                "Chief complaint: Cardiac arrest in parking lot.",
                "Chief complaint: Pneumonia.",
                "Chief complaint: Collapse at gym, arrest."
            ]
        }
        try:
            pd.DataFrame(sample_data).to_csv(data_path, index=False)
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None
    
    try:
        # Load legacy model
        model, tokenizer = load_ohca_model(model_path)
        
        # Run legacy inference
        new_df = pd.read_csv(data_path)
        results = run_inference(
            model=model,
            tokenizer=tokenizer,
            inference_df=new_df,
            output_path="./legacy_results.csv",
            probability_threshold=0.5
        )
        
        # Legacy analysis
        analysis = analyze_predictions(results)
        
        print(f"\nLegacy Method Limitations:")
        print("   - Uses static threshold (0.5) instead of optimal")
        print("   - Less sophisticated confidence categories")
        print("   - No clinical priority guidance")
        print("   - Missing enhanced decision support")
        print(f"\nRecommendation: Upgrade to v3.0 methodology for better performance!")
        
        return results
        
    except Exception as e:
        print(f"Error in legacy inference: {e}")
        return None

def comparison_example():
    """Example comparing v3.0 vs legacy methods side-by-side"""
    
    print("v3.0 vs Legacy Comparison Example")
    print("=" * 40)
    
    # Check what models we have available
    v3_model_path = "./trained_ohca_model_v3"
    legacy_model_path = "./trained_ohca_model"
    
    v3_status = check_model_version(v3_model_path)
    legacy_status = check_model_version(legacy_model_path)
    
    if v3_status == 'not_found' and legacy_status == 'not_found':
        print("No trained models found for comparison")
        print("Train models using both methodologies to see the comparison")
        return None
    
    # Prepare comparison data
    comparison_data = {
        'hadm_id': ['COMP_001', 'COMP_002', 'COMP_003'],
        'clean_text': [
            "Chief complaint: Cardiac arrest at home. Family called 911, CPR initiated immediately.",
            "Chief complaint: Chest pain. Acute MI treated with PCI, stable course, no arrest occurred.", 
            "Chief complaint: Found down at work. Witnessed collapse, coworker CPR, AED shock delivered."
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    print("\nComparison Results:")
    print("-" * 25)
    
    try:
        if v3_status == 'v3.0':
            print("v3.0 Method (with optimal threshold):")
            model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(v3_model_path)
            v3_results = run_inference_with_optimal_threshold(
                model, tokenizer, comp_df, optimal_threshold, output_path=None
            )
            
            for _, row in v3_results.iterrows():
                priority = row.get('clinical_priority', 'N/A')
                print(f"   {row['hadm_id']}: p={row['ohca_probability']:.3f}, "
                      f"pred={row['ohca_prediction']}, priority={priority}")
        
        if legacy_status != 'not_found':
            print("\nLegacy Method (static threshold 0.5):")
            model, tokenizer = load_ohca_model(legacy_model_path)
            legacy_results = run_inference(
                model, tokenizer, comp_df, output_path=None, probability_threshold=0.5
            )
            
            for _, row in legacy_results.iterrows():
                pred = row.get('prediction_050', 'N/A')
                conf = row.get('confidence_category', 'N/A')
                print(f"   {row['hadm_id']}: p={row['ohca_probability']:.3f}, "
                      f"pred={pred}, conf={conf}")
        
        print(f"\nKey Differences:")
        print("   v3.0: Uses optimal threshold, clinical priorities, enhanced workflow")
        print("   Legacy: Static threshold, basic confidence levels, limited guidance")
        
    except Exception as e:
        print(f"Error in comparison: {e}")

def test_model_example():
    """Example of testing model on sample texts"""
    
    print("Model Testing Example")
    print("=" * 25)
    
    # Check available models
    v3_path = "./trained_ohca_model_v3"
    legacy_path = "./trained_ohca_model"
    
    v3_status = check_model_version(v3_path)
    legacy_status = check_model_version(legacy_path)
    
    if v3_status == 'not_found' and legacy_status == 'not_found':
        print("No trained models found for testing")
        return None
    
    # Test samples
    test_samples = {
        'TEST_001': "Cardiac arrest at home, CPR by family",
        'TEST_002': "Chest pain, no arrest, stable course",
        'TEST_003': "Found down at work, immediate CPR given"
    }
    
    # Use best available model
    model_path = v3_path if v3_status == 'v3.0' else legacy_path
    
    try:
        results = test_model_on_sample(model_path, test_samples)
        print("Test completed successfully")
        return results
    except Exception as e:
        print(f"Error in model testing: {e}")
        return None

def batch_processing_v3_example():
    """Example of v3.0 batch processing with optimal threshold"""
    
    print("v3.0 Large Dataset Processing Example")
    print("=" * 45)
    
    model_path = "./trained_ohca_model_v3"
    
    # Check for v3.0 model
    if check_model_version(model_path) != 'v3.0':
        print("v3.0 model not found. Cannot demonstrate batch processing with optimal threshold.")
        return None
    
    # Create sample large dataset
    large_data_path = "large_sample_v3.csv"
    if not os.path.exists(large_data_path):
        print("Creating sample large dataset...")
        
        try:
            # Create 1000 sample records for demonstration
            large_sample = {
                'hadm_id': [f'BATCH_{i:06d}' for i in range(1000)],
                'clean_text': [
                    "Chief complaint: Cardiac arrest at home, bystander CPR initiated.",
                    "Chief complaint: Chest pain, ruled out for MI, no arrest.",
                    "Chief complaint: Found down at work, witnessed cardiac arrest.",
                    "Chief complaint: Shortness of breath, CHF exacerbation treated.",
                    "Chief complaint: Syncope episode, no cardiac arrest occurred.",
                ] * 200  # Repeat to get 1000 samples
            }
            
            pd.DataFrame(large_sample).to_csv(large_data_path, index=False)
            print(f"Sample large dataset created: {large_data_path}")
        except Exception as e:
            print(f"Error creating large dataset: {e}")
            return None
    
    # Process with v3.0 optimal threshold
    print(f"\nProcessing large dataset with v3.0 methodology...")
    
    try:
        result_path = process_large_dataset_with_optimal_threshold(
            model_path=model_path,
            data_path=large_data_path,
            output_path="./large_v3_results.csv",
            chunk_size=200,  # Smaller chunks for demo
            batch_size=16
        )
        
        print(f"v3.0 batch processing complete: {result_path}")
        
        # Analyze batch results
        if os.path.exists(result_path):
            batch_results = pd.read_csv(result_path)
            
            print(f"\nBatch Processing Results:")
            print(f"   Total processed: {len(batch_results):,}")
            print(f"   OHCA detected: {batch_results['ohca_prediction'].sum():,}")
            
            if 'clinical_priority' in batch_results.columns:
                immediate_count = (batch_results['clinical_priority'] == 'Immediate Review').sum()
                print(f"   Immediate review: {immediate_count:,}")
            
            if 'optimal_threshold_used' in batch_results.columns:
                threshold = batch_results['optimal_threshold_used'].iloc[0]
                print(f"   Optimal threshold used: {threshold:.3f}")
        
        return result_path
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='OHCA Inference Examples v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available examples:
  1. v3.0 Inference with Optimal Threshold (RECOMMENDED)
  2. Quick v3.0 Inference
  3. Legacy Inference (backward compatibility)
  4. v3.0 vs Legacy Comparison
  5. v3.0 Batch Processing
  6. Test model on sample texts
        """
    )
    
    parser.add_argument('--example', type=int, choices=range(1, 7), default=1,
                       help='Example to run (1-6, default: 1)')
    parser.add_argument('--model', help='Path to model (optional)')
    parser.add_argument('--data', help='Path to data file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set model paths based on arguments or defaults
    v3_model = args.model if args.model else "./trained_ohca_model_v3"
    legacy_model = args.model if args.model else "./trained_ohca_model"
    data_file = args.data if args.data else "sample_new_data_v3.csv"
    
    try:
        if args.example == 1:
            result = improved_inference_example(v3_model, data_file)
        elif args.example == 2:
            result = quick_inference_v3_example(v3_model, data_file)
        elif args.example == 3:
            result = legacy_inference_example(legacy_model)
        elif args.example == 4:
            result = comparison_example()
        elif args.example == 5:
            result = batch_processing_v3_example()
        elif args.example == 6:
            result = test_model_example()
        
        if result is not None and args.verbose:
            print(f"\nVerbose output:")
            if hasattr(result, 'shape'):
                print(f"Result shape: {result.shape}")
            if hasattr(result, 'columns'):
                print(f"Result columns: {list(result.columns)}")
        
        return 0
        
    except Exception as e:
        print(f"Example failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # If run without arguments, provide interactive menu
    if len(sys.argv) == 1:
        print("OHCA Inference Examples v3.0 - Enhanced Methodology")
        print("=" * 55)
        
        print("\nAvailable examples:")
        print("1. v3.0 Inference with Optimal Threshold (RECOMMENDED)")
        print("2. Quick v3.0 Inference") 
        print("3. Legacy Inference (backward compatibility)")
        print("4. v3.0 vs Legacy Comparison")
        print("5. v3.0 Batch Processing")
        print("6. Test model on sample texts")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                improved_inference_example()
            elif choice == "2":
                quick_inference_v3_example()
            elif choice == "3":
                legacy_inference_example()
            elif choice == "4":
                comparison_example()
            elif choice == "5":
                batch_processing_v3_example()
            elif choice == "6":
                test_model_example()
            else:
                print("Running v3.0 inference example by default...")
                improved_inference_example()
                
        except KeyboardInterrupt:
            print("\nExample cancelled by user")
        except Exception as e:
            print(f"Example failed: {e}")
    else:
        # Use command-line arguments
        sys.exit(main())
