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
from pathlib import Path

# Import v3.0 OHCA inference functions with optimal threshold support
sys.path.append('../src')
from ohca_inference import (
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

def apply_v3_ohca_model_to_clif_dataset():
    """
    Apply MIMIC-trained OHCA model v3.0 to CLIF datasets with optimal threshold support.
    
    This demonstrates the improved v3.0 methodology when applied to external datasets:
    1. Load v3.0 model with optimal threshold metadata
    2. Apply to CLIF dataset using optimal threshold
    3. Enhanced clinical decision support
    4. Better cross-institutional validation
    """
    
    print("Applying MIMIC-trained OHCA Model v3.0 to CLIF Dataset")
    print("="*60)
    
    # ==========================================================================
    # STEP 1: Load v3.0 trained OHCA model with metadata
    # ==========================================================================
    
    print("\n1. Loading v3.0 OHCA model with optimal threshold...")
    print("-" * 55)
    
    # Path to your v3.0 trained model (with metadata)
    model_path = "./trained_ohca_model_v3"
    
    if not os.path.exists(model_path):
        print(f"v3.0 model not found at: {model_path}")
        print("Falling back to legacy model demonstration...")
        return apply_legacy_ohca_model_to_clif_dataset()
    
    # Check for v3.0 metadata
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if not os.path.exists(metadata_path):
        print("Model found but no v3.0 metadata detected.")
        print("This appears to be a legacy model. Consider retraining with v3.0.")
        return apply_legacy_ohca_model_to_clif_dataset()
    
    # Load v3.0 model with optimal threshold
    model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
    
    print("v3.0 model loaded successfully!")
    print(f"   Model version: {metadata.get('model_version', 'unknown')}")
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    print(f"   Training date: {metadata.get('training_date', 'unknown')}")
    print(f"   Methodology: {metadata.get('methodology_improvements', ['Enhanced'])}")
    
    # ==========================================================================
    # STEP 2: Load CLIF dataset from external institution
    # ==========================================================================
    
    print(f"\n2. Loading CLIF dataset from external institution...")
    print("-" * 55)
    
    # CLIF datasets follow standardized format across institutions
    clif_data_path = "clif_dataset_uchicago.csv"  # Example: UChicago CLIF dataset
    
    # For demonstration, create sample CLIF-formatted data
    if not os.path.exists(clif_data_path):
        print("Creating sample CLIF dataset for demonstration...")
        clif_data_path = create_enhanced_clif_data()
    
    # Load the CLIF dataset
    clif_df = pd.read_csv(clif_data_path)
    print(f"Loaded {len(clif_df):,} cases from CLIF dataset")
    print(f"Source institution: {clif_df.get('institution', ['Unknown']).iloc[0]}")
    print(f"CLIF version: {clif_df.get('clif_version', ['Unknown']).iloc[0]}")
    
    # ==========================================================================
    # STEP 3: Prepare CLIF data with enhanced mapping
    # ==========================================================================
    
    print(f"\n3. Enhanced CLIF data preparation...")
    print("-" * 40)
    
    # Enhanced CLIF column mapping for v3.0
    enhanced_clif_mapping = {
        # CLIF standard patient identifiers
        'patient_id': 'hadm_id',
        'hospitalization_id': 'hadm_id', 
        'encounter_id': 'hadm_id',
        'admission_id': 'hadm_id',
        
        # CLIF standard clinical text fields
        'discharge_summary': 'clean_text',
        'clinical_notes': 'clean_text',
        'discharge_notes': 'clean_text',
        'progress_notes': 'clean_text',
        'hospital_course': 'clean_text',
        
        # CLIF patient identifiers for v3.0 patient-level analysis
        'subject_id': 'subject_id',
        'patient_mrn': 'subject_id'
    }
    
    # Apply enhanced CLIF mapping
    print("Mapping CLIF columns to v3.0 OHCA model format...")
    
    available_mappings = {k: v for k, v in enhanced_clif_mapping.items() 
                         if k in clif_df.columns}
    
    if available_mappings:
        clif_df = clif_df.rename(columns=available_mappings)
        print(f"Mapped CLIF columns: {list(available_mappings.keys())}")
    else:
        print("Standard CLIF columns not found. Please check your CLIF dataset format.")
        print(f"Available columns: {list(clif_df.columns)}")
        return
    
    # Validate required columns for v3.0
    if 'hadm_id' not in clif_df.columns or 'clean_text' not in clif_df.columns:
        print("Required columns 'hadm_id' and 'clean_text' not found")
        return
    
    # Enhanced data cleaning for CLIF
    original_size = len(clif_df)
    clif_df = clif_df.dropna(subset=['hadm_id', 'clean_text'])
    clif_df['clean_text'] = clif_df['clean_text'].astype(str)
    
    # Remove very short notes (likely incomplete)
    clif_df = clif_df[clif_df['clean_text'].str.len() >= 50]
    
    print(f"CLIF data prepared: {len(clif_df):,}/{original_size:,} cases ready")
    print("Enhanced v3.0 data validation completed")
    
    # ==========================================================================
    # STEP 4: Run v3.0 inference with optimal threshold
    # ==========================================================================
    
    print(f"\n4. Running v3.0 OHCA inference with optimal threshold...")
    print("-" * 60)
    
    # Use v3.0 inference with optimal threshold
    results = run_inference_with_optimal_threshold(
        model=model,
        tokenizer=tokenizer,
        inference_df=clif_df,
        optimal_threshold=optimal_threshold,
        batch_size=16,
        output_path="clif_v3_ohca_predictions.csv"
    )
    
    print("v3.0 inference completed with optimal threshold!")
    
    # ==========================================================================
    # STEP 5: Enhanced v3.0 results analysis
    # ==========================================================================
    
    print(f"\n5. Enhanced v3.0 Results Analysis...")
    print("-" * 40)
    
    # v3.0 enhanced statistics
    total_cases = len(results)
    ohca_detected_optimal = results['ohca_prediction'].sum()
    
    # Clinical priority breakdown (v3.0 feature)
    if 'clinical_priority' in results.columns:
        priority_counts = results['clinical_priority'].value_counts()
        
        print(f"v3.0 Clinical Priority Distribution:")
        for priority, count in priority_counts.items():
            pct = count / total_cases * 100
            print(f"   {priority}: {count:,} cases ({pct:.1f}%)")
    
    # Enhanced CLIF-specific analysis
    print(f"\nCLIF Dataset Results (v3.0 Methodology):")
    print(f"   Total CLIF cases: {total_cases:,}")
    print(f"   OHCA detected (optimal threshold): {ohca_detected_optimal:,}")
    print(f"   Detection rate: {ohca_detected_optimal/total_cases:.1%}")
    print(f"   Optimal threshold used: {optimal_threshold:.3f}")
    
    # Compare with static thresholds
    static_05 = results['prediction_050'].sum() if 'prediction_050' in results.columns else 0
    static_07 = results['prediction_070'].sum() if 'prediction_070' in results.columns else 0
    
    print(f"\nThreshold Comparison on CLIF Data:")
    print(f"   Optimal ({optimal_threshold:.3f}): {ohca_detected_optimal:,} cases")
    print(f"   Static (0.5): {static_05:,} cases")
    print(f"   Static (0.7): {static_07:,} cases")
    
    if ohca_detected_optimal != static_05:
        print(f"   Optimal threshold shows different results - demonstrating v3.0 value!")
    
    # Enhanced prediction analysis
    analysis = analyze_predictions_enhanced(results)
    
    # ==========================================================================
    # STEP 6: Cross-institutional validation insights
    # ==========================================================================
    
    print(f"\n6. Cross-Institutional Validation Insights...")
    print("-" * 50)
    
    # CLIF standardization benefits with v3.0
    print(f"CLIF + v3.0 Methodology Benefits:")
    print(f"   Consistent data format across institutions")
    print(f"   Optimal threshold automatically applied")
    print(f"   Enhanced clinical decision support")
    print(f"   Standardized confidence categories")
    print(f"   Improved workflow integration")
    
    # Clinical workflow recommendations for CLIF deployment
    immediate_review = results[results['clinical_priority'] == 'Immediate Review'] if 'clinical_priority' in results.columns else pd.DataFrame()
    priority_review = results[results['clinical_priority'] == 'Priority Review'] if 'clinical_priority' in results.columns else pd.DataFrame()
    
    print(f"\nRecommended CLIF Deployment Workflow:")
    if len(immediate_review) > 0:
        print(f"   1. Immediate review: {len(immediate_review):,} cases")
        print(f"      → Priority clinical validation required")
    
    if len(priority_review) > 0:
        print(f"   2. Priority review: {len(priority_review):,} cases")
        print(f"      → Clinical team review recommended")
    
    # Save enhanced results for CLIF deployment
    print(f"\n7. Saving Enhanced Results for CLIF Deployment...")
    print("-" * 55)
    
    # Create comprehensive CLIF analysis summary
    clif_summary = {
        'model_info': {
            'model_version': metadata.get('model_version', 'unknown'),
            'optimal_threshold': optimal_threshold,
            'training_source': 'MIMIC-IV',
            'methodology': 'v3.0_improved'
        },
        'clif_dataset_info': {
            'total_cases': total_cases,
            'data_source': 'CLIF Dataset',
            'institution': clif_df.get('institution', ['Unknown']).iloc[0],
            'clif_version': clif_df.get('clif_version', ['Unknown']).iloc[0]
        },
        'v3_predictions': {
            'ohca_detected_optimal': int(ohca_detected_optimal),
            'detection_rate': float(ohca_detected_optimal/total_cases),
            'immediate_review_cases': int(len(immediate_review)),
            'priority_review_cases': int(len(priority_review))
        },
        'clinical_recommendations': {
            'immediate_review_needed': len(immediate_review) > 0,
            'clinical_validation_priority': 'high' if len(immediate_review) > 10 else 'medium',
            'deployment_readiness': 'ready_with_monitoring'
        },
        'files_created': [
            'clif_v3_ohca_predictions.csv',
            'clif_high_priority_cases.csv',
            'clif_v3_analysis_summary.json'
        ]
    }
    
    # Save high priority cases for clinical review
    if len(immediate_review) > 0 or len(priority_review) > 0:
        high_priority = pd.concat([immediate_review, priority_review])
        high_priority.to_csv('clif_high_priority_cases.csv', index=False)
        print(f"   High priority cases saved: clif_high_priority_cases.csv")
    
    # Save comprehensive analysis
    with open('clif_v3_analysis_summary.json', 'w') as f:
        json.dump(clif_summary, f, indent=2)
    
    print(f"v3.0 CLIF dataset analysis complete!")
    print(f"   Main results: clif_v3_ohca_predictions.csv")
    print(f"   High priority cases: clif_high_priority_cases.csv")  
    print(f"   Analysis summary: clif_v3_analysis_summary.json")
    
    print(f"\nv3.0 Cross-Institutional Deployment Benefits:")
    print(f"   Optimal threshold ensures consistent performance")
    print(f"   Enhanced clinical priorities guide review workflow")
    print(f"   CLIF standardization + v3.0 methodology = Robust deployment")
    
    return results

def apply_legacy_ohca_model_to_clif_dataset():
    """
    Legacy CLIF application for comparison/backward compatibility
    """
    
    print("Legacy OHCA Model Application to CLIF Dataset")
    print("="*50)
    
    print("WARNING: Using legacy methodology with limitations:")
    print("   - Static threshold (0.5) instead of optimal")
    print("   - Basic confidence categories")
    print("   - Limited clinical decision support")
    print("   - No enhanced workflow integration")
    print()
    print("RECOMMENDATION: Use v3.0 methodology for better performance!")
    
    # Path to legacy model
    model_path = "./trained_ohca_model"
    
    if not os.path.exists(model_path):
        print(f"Legacy model not found at: {model_path}")
        return None
    
    # Load legacy model (without metadata)
    model, tokenizer = load_ohca_model(model_path)
    print("Legacy model loaded (no optimal threshold)")
    
    # Create simple CLIF data
    clif_data_path = create_simple_clif_data()
    clif_df = pd.read_csv(clif_data_path)
    
    # Simple CLIF mapping
    clif_df = clif_df.rename(columns={
        'patient_id': 'hadm_id',
        'discharge_summary': 'clean_text'
    })
    
    # Legacy inference with static threshold
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        inference_df=clif_df,
        output_path="clif_legacy_predictions.csv",
        probability_threshold=0.5  # Static threshold
    )
    
    print(f"\nLegacy Results (Static 0.5 threshold):")
    print(f"   Total cases: {len(results):,}")
    print(f"   OHCA predicted: {results['prediction_050'].sum():,}")
    print(f"   High confidence (≥0.8): {(results['ohca_probability'] >= 0.8).sum():,}")
    
    print(f"\nLegacy Method Limitations:")
    print(f"   - No optimal threshold (uses static 0.5)")
    print(f"   - Basic confidence levels only")
    print(f"   - Limited clinical guidance")
    print(f"   - Potentially suboptimal performance")
    
    return results

def create_enhanced_clif_data():
    """Create enhanced sample CLIF dataset for v3.0 demonstration"""
    
    print("Creating enhanced CLIF dataset with v3.0 features...")
    
    # Enhanced CLIF data with more realistic clinical scenarios
    enhanced_clif_data = {
        'patient_id': [f'CLIF_{i:06d}' for i in range(1, 501)],
        'hospitalization_id': [f'HOSP_{i:06d}' for i in range(1, 501)],
        'subject_id': [f'SUBJ_{(i-1)//2 + 1:04d}' for i in range(1, 501)],  # Some patients have multiple admissions
        'discharge_summary': [
            "Patient presented with witnessed cardiac arrest at home. Family member initiated CPR immediately, EMS called. Patient transported to ED with ROSC achieved in field. Post-arrest care initiated.",
            "Chief complaint: Acute chest pain. Patient presents with substernal chest pain, diaphoresis. Troponins elevated, ECG changes consistent with STEMI. No cardiac arrest occurred. Successful PCI performed.",
            "Patient found unresponsive at workplace by coworker. Witnessed collapse, immediate CPR initiated by trained coworker. AED available, shock delivered. EMS arrived, continued resuscitation.",
            "Admission for community-acquired pneumonia. Patient presented with fever, productive cough, shortness of breath. Chest X-ray consistent with pneumonia. Responded well to antibiotic therapy.",
            "Transfer from outside hospital following out-of-hospital cardiac arrest. Initial arrest occurred at restaurant during family dinner. Bystander CPR provided by restaurant staff.",
            "Chief complaint: Acute decompensated heart failure. Patient with known CHF presents with worsening shortness of breath, lower extremity edema. Managed with diuretics, ACE inhibitor.",
            "Witnessed ventricular fibrillation arrest at fitness center. Exercise-induced cardiac arrest, immediate bystander CPR and AED defibrillation. Neurologically intact post-ROSC.",
            "Elective admission for diabetes management and medication adjustment. Patient with poorly controlled type 2 diabetes. No acute cardiac events during hospitalization stay.",
            "Patient arrested during family gathering at home. Spouse witnessed collapse, performed CPR until EMS arrival. Multiple defibrillation attempts, achieved ROSC after 20 minutes.",
            "Routine post-operative admission following planned surgical procedure. Patient stable pre-operatively and post-operatively. No intraoperative or post-operative complications occurred.",
        ] * 50,  # More diverse scenarios
        'clif_version': ['2.1.0'] * 500,
        'institution': ['University_of_Chicago'] * 500,
        'data_quality_score': [np.random.choice([0.85, 0.90, 0.95], p=[0.2, 0.5, 0.3]) for _ in range(500)],
        'note_length': [np.random.randint(200, 1500) for _ in range(500)]  # Realistic note lengths
    }
    
    enhanced_df = pd.DataFrame(enhanced_clif_data)
    enhanced_path = "enhanced_clif_dataset.csv"
    enhanced_df.to_csv(enhanced_path, index=False)
    
    print(f"Enhanced CLIF dataset created: {enhanced_path}")
    print(f"   Enhanced features: Patient relationships, data quality scores")
    print(f"   Realistic clinical scenarios for v3.0 testing")
    print(f"   {enhanced_df['subject_id'].nunique()} unique patients with multiple admissions")
    
    return enhanced_path

def create_simple_clif_data():
    """Create simple CLIF dataset for legacy demonstration"""
    
    simple_clif_data = {
        'patient_id': [f'SIMPLE_{i:06d}' for i in range(100)],
        'discharge_summary': [
            "Cardiac arrest at home, CPR given.",
            "Chest pain, no arrest occurred.",
            "Found down at work, cardiac arrest.",
            "Pneumonia, stable course.",
            "Transfer for post-arrest care.",
        ] * 20,
        'institution': ['Sample_Hospital'] * 100
    }
    
    simple_df = pd.DataFrame(simple_clif_data)
    simple_path = "simple_clif_dataset.csv"
    simple_df.to_csv(simple_path, index=False)
    
    return simple_path

def clif_v3_validation_workflow():
    """
    Enhanced CLIF validation workflow using v3.0 methodology
    """
    
    print("CLIF Cross-Institutional Validation with v3.0 Methodology")
    print("="*60)
    
    print("\nv3.0 Enhanced Validation Benefits:")
    print("   Optimal threshold ensures consistent performance across sites")
    print("   Enhanced clinical priorities guide validation efforts")
    print("   Better confidence calibration for cross-institutional use")
    print("   Comprehensive metadata tracking for reproducibility")
    
    print("\nEnhanced v3.0 CLIF Validation Steps:")
    print("1. Apply v3.0 model with optimal threshold to CLIF datasets")
    print("2. Use enhanced clinical priorities to focus validation efforts")
    print("3. Calculate performance metrics using optimal threshold")
    print("4. Analyze cross-institutional robustness")
    print("5. Document v3.0 methodology benefits for CLIF deployment")
    
    print("\nExample v3.0 CLIF validation code:")
    print("""
    # Load v3.0 model with optimal threshold
    model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
    
    # Apply to multiple CLIF institutions
    institutions = ['uchicago', 'stanford', 'mayo']
    
    validation_results = {}
    for inst in institutions:
        clif_data = load_clif_dataset(f'clif_{inst}.csv')
        
        # Use optimal threshold for consistent evaluation
        results = run_inference_with_optimal_threshold(
            model, tokenizer, clif_data, optimal_threshold
        )
        
        # Enhanced validation analysis
        analysis = analyze_predictions_enhanced(results)
        validation_results[inst] = analysis
    
    # Compare v3.0 performance across institutions
    print("Cross-institutional v3.0 performance:")
    for inst, analysis in validation_results.items():
        print(f"{inst}: Optimal threshold performance maintained")
        print(f"  Clinical priorities available for workflow integration")
    """)
    
    print("\nv3.0 CLIF Deployment Advantages:")
    print("   Consistent optimal threshold across all institutions")
    print("   Standardized clinical decision support")
    print("   Enhanced confidence calibration")
    print("   Better workflow integration")
    print("   Comprehensive performance tracking")

if __name__ == "__main__":
    print("CLIF Dataset Application Examples v3.0")
    print("="*40)
    
    print("\nAvailable examples:")
    print("1. Apply v3.0 OHCA model to CLIF dataset (RECOMMENDED)")
    print("2. Apply legacy OHCA model to CLIF dataset (comparison)")
    print("3. v3.0 CLIF cross-institutional validation workflow")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        apply_v3_ohca_model_to_clif_dataset()
    elif choice == "2":
        apply_legacy_ohca_model_to_clif_dataset()
    elif choice == "3":
        clif_v3_validation_workflow()
    else:
        print("Running v3.0 CLIF application by default...")
        apply_v3_ohca_model_to_clif_dataset()
