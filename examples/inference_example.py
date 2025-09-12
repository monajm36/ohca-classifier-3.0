"""
OHCA Inference Example v3.0 - Enhanced with Optimal Threshold Support

This example shows how to use pre-trained OHCA classifiers with the improved
v3.0 methodology, including optimal threshold usage and enhanced clinical 
decision support.
"""

import pandas as pd
import sys
import os

# Add src to path for development
sys.path.append('../src')

# v3.0 imports - improved functions with optimal threshold support
from ohca_inference import (
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

def improved_inference_example():
    """Example using v3.0 methodology with optimal threshold (RECOMMENDED)"""
    
    print("🚀 OHCA Inference v3.0 - Improved Methodology Example")
    print("="*60)
    
    # ==========================================================================
    # STEP 1: Check for v3.0 model with metadata
    # ==========================================================================
    
    model_path = "./trained_ohca_model_v3"  # v3.0 model with metadata
    
    if not os.path.exists(model_path):
        print(f"❌ v3.0 Model not found at: {model_path}")
        print("Please train a model using complete_improved_training_pipeline() first.")
        print("Falling back to legacy example...")
        return legacy_inference_example()
    
    # Check for metadata file
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if not os.path.exists(metadata_path):
        print("⚠️  Model found but no metadata detected. This appears to be a legacy model.")
        print("Consider retraining with v3.0 methodology for optimal performance.")
        return legacy_inference_example()
    
    # ==========================================================================
    # STEP 2: Prepare sample data
    # ==========================================================================
    
    new_data_path = "sample_new_data_v3.csv"
    
    if not os.path.exists(new_data_path):
        print("Creating enhanced sample data for v3.0 demonstration...")
        
        sample_data = {
            'hadm_id': [f'V3_{i:06d}' for i in range(1, 21)],
            'clean_text': [
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
        }
        
        new_df = pd.DataFrame(sample_data)
        new_df.to_csv(new_data_path, index=False)
        print(f"✅ Sample data created: {new_data_path}")
    
    # ==========================================================================
    # STEP 3: Load v3.0 model with metadata
    # ==========================================================================
    
    print(f"\n📂 STEP 3: Loading v3.0 Model with Metadata")
    print("-" * 50)
    
    model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model version: {metadata.get('model_version', 'unknown')}")
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    print(f"   Training date: {metadata.get('training_date', 'unknown')}")
    
    # ==========================================================================
    # STEP 4: Run inference with optimal threshold
    # ==========================================================================
    
    print(f"\n🔍 STEP 4: Running Inference with Optimal Threshold")
    print("-" * 55)
    
    new_df = pd.read_csv(new_data_path)
    print(f"Loaded {len(new_df)} cases for inference")
    
    # Run enhanced inference with optimal threshold
    results = run_inference_with_optimal_threshold(
        model=model,
        tokenizer=tokenizer,
        inference_df=new_df,
        optimal_threshold=optimal_threshold,
        batch_size=8,
        output_path="./v3_inference_results.csv"
    )
    
    # ==========================================================================
    # STEP 5: Analyze enhanced results
    # ==========================================================================
    
    print(f"\n📊 STEP 5: Enhanced Results Analysis")
    print("-" * 45)
    
    analysis = analyze_predictions_enhanced(results)
    
    # Show clinical priorities
    if 'clinical_priority' in results.columns:
        print(f"\n🏥 Clinical Priority Cases:")
        print("-" * 30)
        
        immediate = results[results['clinical_priority'] == 'Immediate Review']
        priority = results[results['clinical_priority'] == 'Priority Review'] 
        clinical = results[results['clinical_priority'] == 'Clinical Review']
        
        if len(immediate) > 0:
            print(f"🔴 Immediate Review ({len(immediate)} cases):")
            for _, row in immediate.iterrows():
                hadm_id = row['hadm_id']
                prob = row['ohca_probability']
                text = new_df[new_df['hadm_id'] == hadm_id]['clean_text'].iloc[0]
                print(f"   {hadm_id}: p={prob:.3f} - {text[:80]}...")
        
        if len(priority) > 0:
            print(f"\n🟡 Priority Review ({len(priority)} cases):")
            for _, row in priority.head(3).iterrows():  # Show first 3
                hadm_id = row['hadm_id']
                prob = row['ohm_probability']
                print(f"   {hadm_id}: p={prob:.3f}")
    
    # ==========================================================================
    # STEP 6: Compare with legacy thresholds
    # ==========================================================================
    
    print(f"\n⚖️  STEP 6: Threshold Comparison")
    print("-" * 35)
    
    optimal_detections = results['ohca_prediction'].sum()
    static_050_detections = results['prediction_050'].sum()
    static_070_detections = results['prediction_070'].sum()
    
    print(f"Optimal threshold ({optimal_threshold:.3f}): {optimal_detections} OHCA cases")
    print(f"Static threshold (0.5): {static_050_detections} OHCA cases")
    print(f"Static threshold (0.7): {static_070_detections} OHCA cases")
    
    if optimal_detections != static_050_detections:
        print(f"✅ Optimal threshold shows different results than static 0.5!")
        print(f"   This demonstrates the value of threshold optimization.")
    
    # ==========================================================================
    # STEP 7: Clinical workflow integration
    # ==========================================================================
    
    print(f"\n👩‍⚕️ STEP 7: Clinical Workflow Integration")
    print("-" * 45)
    
    print("Recommended workflow based on v3.0 results:")
    print("1. Immediate Review cases → Priority manual review")
    print("2. Priority Review cases → Clinical team review")  
    print("3. Clinical Review cases → Consider for quality checks")
    print("4. Lower priority cases → Routine processing")
    
    # Show expected clinical impact
    total_cases = len(results)
    high_priority_cases = len(results[results['clinical_priority'].isin(['Immediate Review', 'Priority Review'])])
    
    if high_priority_cases > 0:
        efficiency_gain = (total_cases - high_priority_cases) / total_cases * 100
        print(f"\n📈 Expected Efficiency Gains:")
        print(f"   Focus review on {high_priority_cases}/{total_cases} cases ({high_priority_cases/total_cases*100:.1f}%)")
        print(f"   Potential {efficiency_gain:.1f}% reduction in manual review burden")
    
    print(f"\n✅ v3.0 INFERENCE COMPLETE!")
    print("="*50)
    print("Key v3.0 advantages demonstrated:")
    print("✅ Optimal threshold from validation set")
    print("✅ Enhanced clinical decision support")
    print("✅ Improved confidence categorization")
    print("✅ Better workflow integration")
    
    return results

def quick_inference_v3_example():
    """Quick inference using v3.0 convenience function (RECOMMENDED)"""
    
    print("⚡ Quick v3.0 Inference Example")
    print("="*35)
    
    model_path = "./trained_ohca_model_v3"
    data_path = "sample_new_data_v3.csv"
    
    # Check if we have a v3.0 model
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_path):
        print("✅ Detected v3.0 model - using optimal threshold")
        
        # Use the improved quick inference function
        results = quick_inference_with_optimal_threshold(
            model_path=model_path,
            data_path=data_path,
            output_path="./quick_v3_results.csv"
        )
        
        print(f"\n🎯 v3.0 Quick Results:")
        print(f"   Optimal threshold used: {results['optimal_threshold_used'].iloc[0]:.3f}")
        print(f"   OHCA detected: {results['ohca_prediction'].sum()}")
        print(f"   Immediate review needed: {(results['clinical_priority'] == 'Immediate Review').sum()}")
        
    else:
        print("⚠️  No v3.0 model found - falling back to legacy method")
        results = quick_inference(
            model_path=model_path,
            data_path=data_path,
            output_path="./quick_legacy_results.csv"
        )
    
    return results

def legacy_inference_example():
    """Legacy inference example for backward compatibility"""
    
    print("🔄 Legacy Inference Example (Backward Compatibility)")
    print("="*55)
    
    model_path = "./trained_ohca_model"  # Legacy model without metadata
    
    if not os.path.exists(model_path):
        print(f"❌ Legacy model not found at: {model_path}")
        print("Please train a model first or use the v3.0 methodology.")
        return None
    
    print("ℹ️  Using legacy inference method with static threshold 0.5")
    
    # Create sample data if needed
    data_path = "sample_legacy_data.csv"
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
        pd.DataFrame(sample_data).to_csv(data_path, index=False)
    
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
    
    print(f"\n⚠️  Legacy Method Limitations:")
    print("   - Uses static threshold (0.5) instead of optimal")
    print("   - Less sophisticated confidence categories")
    print("   - No clinical priority guidance")
    print("   - Missing enhanced decision support")
    print(f"\n💡 Recommendation: Upgrade to v3.0 methodology for better performance!")
    
    return results

def comparison_example():
    """Example comparing v3.0 vs legacy methods side-by-side"""
    
    print("⚖️  v3.0 vs Legacy Comparison Example")
    print("="*40)
    
    # Check what models we have available
    v3_model_path = "./trained_ohca_model_v3"
    legacy_model_path = "./trained_ohca_model"
    
    v3_available = os.path.exists(os.path.join(v3_model_path, 'model_metadata.json'))
    legacy_available = os.path.exists(legacy_model_path)
    
    if not (v3_available or legacy_available):
        print("❌ No trained models found for comparison")
        print("Train models using both methodologies to see the comparison")
        return
    
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
    
    print("\n📊 Comparison Results:")
    print("-" * 25)
    
    if v3_available:
        print("🟢 v3.0 Method (with optimal threshold):")
        model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(v3_model_path)
        v3_results = run_inference_with_optimal_threshold(
            model, tokenizer, comp_df, optimal_threshold, output_path=None
        )
        
        for _, row in v3_results.iterrows():
            print(f"   {row['hadm_id']}: p={row['ohca_probability']:.3f}, "
                  f"pred={row['ohca_prediction']}, priority={row['clinical_priority']}")
    
    if legacy_available:
        print("\n🔴 Legacy Method (static threshold 0.5):")
        model, tokenizer = load_ohca_model(legacy_model_path)
        legacy_results = run_inference(
            model, tokenizer, comp_df, output_path=None, probability_threshold=0.5
        )
        
        for _, row in legacy_results.iterrows():
            print(f"   {row['hadm_id']}: p={row['ohca_probability']:.3f}, "
                  f"pred={row['prediction_050']}, conf={row['confidence_category']}")
    
    print(f"\n📈 Key Differences:")
    print("   v3.0: Uses optimal threshold, clinical priorities, enhanced workflow")
    print("   Legacy: Static threshold, basic confidence levels, limited guidance")

def batch_processing_v3_example():
    """Example of v3.0 batch processing with optimal threshold"""
    
    print("📦 v3.0 Large Dataset Processing Example")
    print("="*45)
    
    model_path = "./trained_ohca_model_v3"
    
    # Check for v3.0 model
    if not os.path.exists(os.path.join(model_path, 'model_metadata.json')):
        print("⚠️  v3.0 model not found. Using legacy batch processing...")
        return legacy_batch_processing_example()
    
    # Create sample large dataset
    large_data_path = "large_sample_v3.csv"
    if not os.path.exists(large_data_path):
        print("Creating sample large dataset...")
        
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
        print(f"✅ Sample large dataset created: {large_data_path}")
    
    # Process with v3.0 optimal threshold
    print(f"\n🔄 Processing large dataset with v3.0 methodology...")
    
    result_path = process_large_dataset_with_optimal_threshold(
        model_path=model_path,
        data_path=large_data_path,
        output_path="./large_v3_results.csv",
        chunk_size=200,  # Smaller chunks for demo
        batch_size=16
    )
    
    print(f"✅ v3.0 batch processing complete: {result_path}")
    
    # Analyze batch results
    if os.path.exists(result_path):
        batch_results = pd.read_csv(result_path)
        
        print(f"\n📊 Batch Processing Results:")
        print(f"   Total processed: {len(batch_results):,}")
        print(f"   OHCA detected: {batch_results['ohca_prediction'].sum():,}")
        print(f"   Immediate review: {(batch_results['clinical_priority'] == 'Immediate Review').sum():,}")
        print(f"   Optimal threshold used: {batch_results['optimal_threshold_used'].iloc[0]:.3f}")
    
    return result_path

def legacy_batch_processing_example():
    """Legacy batch processing for comparison"""
    
    print("📦 Legacy Batch Processing (for comparison)")
    print("="*45)
    
    # This would use the legacy process_large_dataset function
    # Implementation similar to original but with warnings about limitations
    
    print("⚠️  Using legacy batch processing method")
    print("   - Static threshold instead of optimal")
    print("   - Basic confidence categories only")
    print("   - Limited clinical decision support")
    
    # Implementation would go here...
    return None

if __name__ == "__main__":
    print("OHCA Inference Examples v3.0 - Enhanced Methodology")
    print("="*55)
    
    print("\nAvailable examples:")
    print("1. v3.0 Inference with Optimal Threshold (RECOMMENDED)")
    print("2. Quick v3.0 Inference") 
    print("3. Legacy Inference (backward compatibility)")
    print("4. v3.0 vs Legacy Comparison")
    print("5. v3.0 Batch Processing")
    print("6. Test model on sample texts")
    
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
        # Test model example works with both v3.0 and legacy
        test_model_on_sample("./trained_ohca_model_v3", {
            'TEST_001': "Cardiac arrest at home, CPR by family",
            'TEST_002': "Chest pain, no arrest, stable course"
        })
    else:
        print("Running v3.0 inference example by default...")
        improved_inference_example()
