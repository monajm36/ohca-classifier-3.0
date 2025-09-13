#!/usr/bin/env python3
"""
Applying OHCA Classifier v3.0 to CLIF Datasets

This example demonstrates how to apply a MIMIC-trained OHCA model with v3.0 
methodology improvements to CLIF datasets from other institutions. CLIF 
(Common Longitudinal ICU data Format) standardizes healthcare data, making 
cross-institutional model deployment much easier.

Key v3.0 improvements:
- Automatic optimal threshold usage
- Enhanced clinical decision support  
- Better confidence categorization
- Improved workflow integration

Example use case: Apply MIMIC-IV trained model → University of Chicago CLIF dataset
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path

# Use proper package imports instead of path manipulation
try:
    # Try to import from installed package first
    from src.ohca_inference import (
        # v3.0 functions (RECOMMENDED)
        load_ohca_model_with_metadata,
        run_inference_with_optimal_threshold,
        quick_inference_with_optimal_threshold,
        analyze_predictions_enhanced,
        
        # Legacy functions (backward compatibility)
        load_ohca_model,
        run_inference,
        analyze_predictions,
        get_high_confidence_cases
    )
except ImportError:
    # Fallback for development environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ohca_inference import (
            load_ohca_model_with_metadata,
            run_inference_with_optimal_threshold,
            quick_inference_with_optimal_threshold,
            analyze_predictions_enhanced,
            load_ohca_model,
            run_inference,
            analyze_predictions,
            get_high_confidence_cases
        )
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Please ensure the package is installed with: pip install -e .")
        sys.exit(1)

def create_sample_clif_data(output_path="sample_clif_dataset.csv", size=100):
    """
    Create realistic sample CLIF dataset for demonstration
    
    Args:
        output_path: Path to save the sample dataset
        size: Number of sample records to create
        
    Returns:
        str: Path to created dataset
    """
    print(f"Creating sample CLIF dataset with {size} records...")
    
    # Realistic OHCA and non-OHCA scenarios for CLIF
    ohca_templates = [
        "Patient presented with witnessed cardiac arrest at home. Family member initiated CPR immediately, EMS called. Patient transported to ED with ROSC achieved in field. Post-arrest care initiated.",
        "Patient found unresponsive at workplace by coworker. Witnessed collapse, immediate CPR initiated by trained coworker. AED available, shock delivered. EMS arrived, continued resuscitation.",
        "Transfer from outside hospital following out-of-hospital cardiac arrest. Initial arrest occurred at restaurant during family dinner. Bystander CPR provided by restaurant staff.",
        "Witnessed ventricular fibrillation arrest at fitness center. Exercise-induced cardiac arrest, immediate bystander CPR and AED defibrillation. Neurologically intact post-ROSC.",
        "Patient arrested during family gathering at home. Spouse witnessed collapse, performed CPR until EMS arrival. Multiple defibrillation attempts, achieved ROSC after 20 minutes.",
        "Out-of-hospital cardiac arrest while shopping. Collapse witnessed by store employees, immediate CPR and AED. EMS response time 6 minutes, sustained ROSC achieved.",
        "Cardiac arrest at home during sleep, found by spouse in morning. CPR initiated immediately, EMS called. Prolonged resuscitation effort, eventual ROSC in emergency department."
    ]
    
    non_ohca_templates = [
        "Chief complaint: Acute chest pain. Patient presents with substernal chest pain, diaphoresis. Troponins elevated, ECG changes consistent with STEMI. No cardiac arrest occurred. Successful PCI performed.",
        "Admission for community-acquired pneumonia. Patient presented with fever, productive cough, shortness of breath. Chest X-ray consistent with pneumonia. Responded well to antibiotic therapy.",
        "Chief complaint: Acute decompensated heart failure. Patient with known CHF presents with worsening shortness of breath, lower extremity edema. Managed with diuretics, ACE inhibitor.",
        "Elective admission for diabetes management and medication adjustment. Patient with poorly controlled type 2 diabetes. No acute cardiac events during hospitalization stay.",
        "Routine post-operative admission following planned surgical procedure. Patient stable pre-operatively and post-operatively. No intraoperative or post-operative complications occurred.",
        "Admission for acute kidney injury. Patient with baseline chronic kidney disease presents with elevated creatinine. Managed conservatively with fluid management and medication adjustment.",
        "Chief complaint: Acute stroke symptoms. Patient presented with sudden onset left-sided weakness and aphasia. CT scan confirms acute ischemic stroke, tissue plasminogen activator administered."
    ]
    
    # Generate balanced dataset
    ohca_count = size // 10  # About 10% OHCA (realistic prevalence)
    non_ohca_count = size - ohca_count
    
    data = []
    
    # Generate OHCA cases
    for i in range(ohca_count):
        template = np.random.choice(ohca_templates)
        data.append({
            'hospitalization_id': f'CLIF_OHCA_{i+1:04d}',
            'patient_id': f'SUBJ_{i+1:04d}',
            'subject_id': f'SUBJ_{i+1:04d}',
            'discharge_summary': template,
            'institution': 'University_of_Chicago',
            'clif_version': '2.1.0',
            'data_quality_score': np.random.uniform(0.85, 0.98),
            'case_type': 'OHCA_example'
        })
    
    # Generate non-OHCA cases
    for i in range(non_ohca_count):
        template = np.random.choice(non_ohca_templates)
        data.append({
            'hospitalization_id': f'CLIF_NON_{i+1:04d}',
            'patient_id': f'SUBJ_{ohca_count + i+1:04d}',
            'subject_id': f'SUBJ_{ohca_count + i+1:04d}',
            'discharge_summary': template,
            'institution': 'University_of_Chicago',
            'clif_version': '2.1.0',
            'data_quality_score': np.random.uniform(0.80, 0.95),
            'case_type': 'non_OHCA_example'
        })
    
    # Shuffle the data
    np.random.shuffle(data)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Sample CLIF dataset created: {output_path}")
        print(f"  Total records: {len(df):,}")
        print(f"  OHCA examples: {ohca_count}")
        print(f"  Non-OHCA examples: {non_ohca_count}")
        print(f"  Institution: University_of_Chicago")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error creating sample dataset: {e}")

def prepare_clif_data(clif_df):
    """
    Prepare CLIF data for OHCA model input
    
    Args:
        clif_df: DataFrame with CLIF data
        
    Returns:
        DataFrame: Prepared data with required columns
    """
    print("Preparing CLIF data for OHCA model...")
    
    # Enhanced CLIF column mapping
    clif_mapping = {
        # Patient identifiers
        'patient_id': 'hadm_id',
        'hospitalization_id': 'hadm_id', 
        'encounter_id': 'hadm_id',
        'admission_id': 'hadm_id',
        
        # Clinical text fields
        'discharge_summary': 'clean_text',
        'clinical_notes': 'clean_text',
        'discharge_notes': 'clean_text',
        'progress_notes': 'clean_text',
        'hospital_course': 'clean_text',
        
        # Patient identifiers for patient-level analysis
        'subject_id': 'subject_id',
        'patient_mrn': 'subject_id'
    }
    
    # Apply mapping for available columns
    available_mappings = {k: v for k, v in clif_mapping.items() if k in clif_df.columns}
    
    if available_mappings:
        clif_df = clif_df.rename(columns=available_mappings)
        print(f"Applied CLIF mappings: {list(available_mappings.keys())}")
    else:
        print("Warning: No standard CLIF columns found for mapping")
    
    # Validate required columns
    required_cols = ['hadm_id', 'clean_text']
    missing_cols = [col for col in required_cols if col not in clif_df.columns]
    
    if missing_cols:
        # Try to help user identify correct columns
        available_cols = list(clif_df.columns)
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {available_cols}")
        raise ValueError(f"Cannot proceed without required columns: {missing_cols}")
    
    # Clean and validate data
    initial_size = len(clif_df)
    clif_df = clif_df.dropna(subset=required_cols)
    clif_df['clean_text'] = clif_df['clean_text'].astype(str)
    clif_df['hadm_id'] = clif_df['hadm_id'].astype(str)
    
    # Remove very short notes
    clif_df = clif_df[clif_df['clean_text'].str.len() >= 20]
    
    final_size = len(clif_df)
    if final_size < initial_size:
        print(f"Cleaned data: {final_size:,}/{initial_size:,} records retained")
    
    if final_size == 0:
        raise ValueError("No valid records remaining after data cleaning")
    
    return clif_df

def analyze_clif_results(results, metadata=None):
    """
    Analyze OHCA prediction results for CLIF dataset
    
    Args:
        results: DataFrame with prediction results
        metadata: Model metadata dict
        
    Returns:
        dict: Analysis summary
    """
    print("\nAnalyzing CLIF prediction results...")
    
    total_cases = len(results)
    
    # Get threshold information
    if 'optimal_threshold_used' in results.columns:
        threshold_used = results['optimal_threshold_used'].iloc[0]
        is_optimal = True
    else:
        threshold_used = 0.5
        is_optimal = False
    
    # Count predictions
    if 'ohca_prediction' in results.columns:
        ohca_detected = results['ohca_prediction'].sum()
    else:
        ohca_detected = (results['ohca_probability'] >= threshold_used).sum()
    
    # Clinical priority analysis (v3.0 feature)
    priority_analysis = {}
    if 'clinical_priority' in results.columns:
        priority_counts = results['clinical_priority'].value_counts()
        priority_analysis = priority_counts.to_dict()
    
    # Confidence distribution
    high_conf = (results['ohca_probability'] >= 0.8).sum()
    very_high_conf = (results['ohca_probability'] >= 0.9).sum()
    medium_conf = (results['ohca_probability'] >= 0.6).sum() - high_conf
    
    analysis = {
        'total_cases': int(total_cases),
        'ohca_detected': int(ohca_detected),
        'detection_rate': float(ohca_detected / total_cases),
        'threshold_used': float(threshold_used),
        'is_optimal_threshold': is_optimal,
        'confidence_distribution': {
            'very_high_conf_90+': int(very_high_conf),
            'high_conf_80_90': int(high_conf - very_high_conf),
            'medium_conf_60_80': int(medium_conf),
            'lower_conf': int(total_cases - medium_conf - high_conf)
        },
        'clinical_priorities': priority_analysis,
        'model_info': metadata or {}
    }
    
    # Print summary
    print(f"CLIF Dataset Analysis Results:")
    print(f"  Total cases: {total_cases:,}")
    print(f"  OHCA detected: {ohca_detected:,} ({analysis['detection_rate']:.1%})")
    print(f"  Threshold: {threshold_used:.3f} ({'optimal' if is_optimal else 'default'})")
    
    if priority_analysis:
        print(f"  Clinical Priorities:")
        for priority, count in priority_analysis.items():
            print(f"    {priority}: {count:,}")
    
    return analysis

def apply_ohca_to_clif(model_path, clif_data_path, output_dir="./clif_results"):
    """
    Apply OHCA model to CLIF dataset with comprehensive analysis
    
    Args:
        model_path: Path to trained OHCA model
        clif_data_path: Path to CLIF dataset CSV
        output_dir: Directory to save results
        
    Returns:
        dict: Complete analysis results
    """
    print("Applying OHCA Model v3.0 to CLIF Dataset")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(clif_data_path):
        raise FileNotFoundError(f"CLIF data not found: {clif_data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load model
    print(f"\n1. Loading OHCA model from: {model_path}")
    print("-" * 40)
    
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    is_v3_model = os.path.exists(metadata_path)
    
    if is_v3_model:
        try:
            model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
            print(f"v3.0 model loaded successfully!")
            print(f"  Model version: {metadata.get('model_version', 'unknown')}")
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
        except Exception as e:
            raise RuntimeError(f"Error loading v3.0 model: {e}")
    else:
        try:
            model, tokenizer = load_ohca_model(model_path)
            optimal_threshold = 0.5
            metadata = {'model_version': 'legacy', 'optimal_threshold': 0.5}
            print("Legacy model loaded (using default threshold 0.5)")
        except Exception as e:
            raise RuntimeError(f"Error loading legacy model: {e}")
    
    # Step 2: Load and prepare CLIF data
    print(f"\n2. Loading CLIF dataset from: {clif_data_path}")
    print("-" * 40)
    
    try:
        clif_df = pd.read_csv(clif_data_path)
        print(f"Loaded {len(clif_df):,} records from CLIF dataset")
        
        # Show dataset info if available
        if 'institution' in clif_df.columns:
            institutions = clif_df['institution'].unique()
            print(f"  Institutions: {list(institutions)}")
        
        if 'clif_version' in clif_df.columns:
            versions = clif_df['clif_version'].unique()
            print(f"  CLIF versions: {list(versions)}")
        
    except Exception as e:
        raise RuntimeError(f"Error loading CLIF data: {e}")
    
    # Prepare data
    try:
        prepared_df = prepare_clif_data(clif_df)
    except Exception as e:
        raise RuntimeError(f"Error preparing CLIF data: {e}")
    
    # Step 3: Run inference
    print(f"\n3. Running OHCA inference...")
    print("-" * 30)
    
    results_path = os.path.join(output_dir, "clif_ohca_predictions.csv")
    
    try:
        if is_v3_model:
            print("Using v3.0 inference with optimal threshold...")
            results = run_inference_with_optimal_threshold(
                model=model,
                tokenizer=tokenizer,
                inference_df=prepared_df,
                optimal_threshold=optimal_threshold,
                output_path=results_path
            )
        else:
            print("Using legacy inference with default threshold...")
            results = run_inference(
                model=model,
                tokenizer=tokenizer,
                inference_df=prepared_df,
                output_path=results_path
            )
    except Exception as e:
        raise RuntimeError(f"Error during inference: {e}")
    
    # Step 4: Analyze results
    print(f"\n4. Analyzing results...")
    print("-" * 25)
    
    try:
        analysis = analyze_clif_results(results, metadata)
    except Exception as e:
        print(f"Warning: Error in analysis: {e}")
        analysis = {'error': str(e)}
    
    # Step 5: Save comprehensive results
    print(f"\n5. Saving results to: {output_dir}")
    print("-" * 35)
    
    try:
        # Save analysis summary
        analysis_path = os.path.join(output_dir, "clif_analysis_summary.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save high priority cases if available
        if 'clinical_priority' in results.columns:
            high_priority = results[
                results['clinical_priority'].isin(['Immediate Review', 'Priority Review'])
            ]
            if len(high_priority) > 0:
                high_priority_path = os.path.join(output_dir, "clif_high_priority_cases.csv")
                high_priority.to_csv(high_priority_path, index=False)
                print(f"  High priority cases: {high_priority_path}")
        
        print(f"  Main results: {results_path}")
        print(f"  Analysis summary: {analysis_path}")
        
    except Exception as e:
        print(f"Warning: Error saving additional files: {e}")
    
    # Summary
    print(f"\n6. CLIF Application Summary")
    print("-" * 30)
    print(f"  Model type: {'v3.0' if is_v3_model else 'Legacy'}")
    print(f"  Cases processed: {len(results):,}")
    print(f"  OHCA detected: {analysis.get('ohca_detected', 'N/A'):,}")
    print(f"  Results saved in: {output_dir}")
    
    if is_v3_model:
        print(f"\nv3.0 Benefits Demonstrated:")
        print(f"  ✓ Optimal threshold automatically applied")
        print(f"  ✓ Enhanced clinical priorities provided")
        print(f"  ✓ Better confidence calibration")
        print(f"  ✓ Comprehensive metadata tracking")
    else:
        print(f"\nLegacy Model Limitations:")
        print(f"  - Static threshold (0.5) used")
        print(f"  - Basic confidence levels only")
        print(f"  - Consider upgrading to v3.0 methodology")
    
    return {
        'results': results,
        'analysis': analysis,
        'output_dir': output_dir,
        'model_version': metadata.get('model_version', 'unknown')
    }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Apply OHCA classifier to CLIF datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing CLIF dataset
  python clif_dataset_example.py --model ./my_model --data clif_data.csv
  
  # Create sample data and run analysis
  python clif_dataset_example.py --model ./my_model --create-sample
  
  # Full analysis with custom output directory
  python clif_dataset_example.py --model ./my_model --data clif_data.csv --output ./clif_analysis
        """
    )
    
    parser.add_argument('--model', required=True, help='Path to trained OHCA model')
    parser.add_argument('--data', help='Path to CLIF dataset CSV')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample CLIF dataset for demonstration')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Size of sample dataset (default: 100)')
    parser.add_argument('--output', default='./clif_results',
                       help='Output directory for results (default: ./clif_results)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Train a model first using: ohca-train your_labeled_data.csv")
        return 1
    
    # Handle data source
    if args.create_sample:
        print("Creating sample CLIF dataset...")
        try:
            clif_data_path = create_sample_clif_data(
                "sample_clif_dataset.csv", 
                args.sample_size
            )
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return 1
    elif args.data:
        if not os.path.exists(args.data):
            print(f"Error: CLIF data file not found: {args.data}")
            return 1
        clif_data_path = args.data
    else:
        print("Error: Must specify either --data or --create-sample")
        return 1
    
    # Run analysis
    try:
        result = apply_ohca_to_clif(args.model, clif_data_path, args.output)
        
        if args.verbose:
            print(f"\nDetailed Results:")
            analysis = result['analysis']
            print(f"  Detection rate: {analysis.get('detection_rate', 0):.1%}")
            print(f"  Model version: {result['model_version']}")
            
            if 'confidence_distribution' in analysis:
                print(f"  Confidence distribution:")
                for level, count in analysis['confidence_distribution'].items():
                    print(f"    {level}: {count}")
        
        print(f"\nCLIF analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"CLIF analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # If run directly, provide interactive options
    if len(sys.argv) == 1:
        print("CLIF Dataset Example - OHCA Classifier v3.0")
        print("=" * 45)
        
        print("\nOptions:")
        print("1. Apply v3.0 model with sample CLIF data")
        print("2. Apply model to existing CLIF dataset")
        print("3. Create sample CLIF data only")
        print("4. Interactive command-line mode")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Quick demo with sample data
            model_path = input("Enter model path (or press Enter for './trained_ohca_model_v3'): ").strip()
            if not model_path:
                model_path = "./trained_ohca_model_v3"
            
            try:
                clif_path = create_sample_clif_data()
                apply_ohca_to_clif(model_path, clif_path)
            except Exception as e:
                print(f"Demo failed: {e}")
        
        elif choice == "2":
            model_path = input("Enter model path: ").strip()
            data_path = input("Enter CLIF data path: ").strip()
            
            try:
                apply_ohca_to_clif(model_path, data_path)
            except Exception as e:
                print(f"Analysis failed: {e}")
        
        elif choice == "3":
            size = input("Enter dataset size (default 100): ").strip()
            size = int(size) if size.isdigit() else 100
            
            try:
                create_sample_clif_data(size=size)
            except Exception as e:
                print(f"Sample creation failed: {e}")
        
        elif choice == "4":
            print("\nRun with --help to see command-line options")
            print("Example: python clif_dataset_example.py --help")
        
        else:
            print("Invalid choice")
    else:
        # Use command-line arguments
        sys.exit(main())
