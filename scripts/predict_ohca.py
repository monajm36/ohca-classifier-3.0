#!/usr/bin/env python3
"""
Apply OHCA Classifier to New Discharge Notes

This script applies a trained OHCA classifier to new discharge notes.
Input data should have columns: hadm_id, clean_text

Supports both v3.0 models (with optimal thresholds) and legacy models.
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Use proper package imports instead of path manipulation
try:
    # Try to import from installed package first
    from src.ohca_inference import (
        quick_inference, quick_inference_with_optimal_threshold,
        process_large_dataset_with_optimal_threshold,
        analyze_predictions_enhanced
    )
except ImportError:
    # Fallback for development environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ohca_inference import (
            quick_inference, quick_inference_with_optimal_threshold,
            process_large_dataset_with_optimal_threshold,
            analyze_predictions_enhanced
        )
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Please ensure the package is installed with: pip install -e .")
        sys.exit(1)

def validate_discharge_data(df):
    """Validate that discharge data has required columns"""
    required_cols = ['hadm_id', 'clean_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) == 0:
        raise ValueError("Input data is empty")
    
    # Check for missing values
    missing_ids = df['hadm_id'].isna().sum()
    missing_text = df['clean_text'].isna().sum()
    
    # Count completely empty text fields
    empty_text = df['clean_text'].astype(str).isin(['', 'nan', 'None', 'null']).sum()
    
    if missing_ids > 0:
        print(f"Warning: {missing_ids} rows have missing hadm_id - these will be removed")
    if missing_text > 0:
        print(f"Warning: {missing_text} rows have missing clean_text - these will be removed")
    if empty_text > 0:
        print(f"Warning: {empty_text} rows have empty/null text - these will be handled during inference")
    
    # Count valid records
    valid_records = len(df.dropna(subset=required_cols))
    
    print(f"Data validation:")
    print(f"  Total discharge notes: {len(df):,}")
    print(f"  Valid records: {valid_records:,}")
    
    if valid_records == 0:
        raise ValueError("No valid records found after removing missing values")
    
    # Check for duplicate hadm_ids
    duplicate_ids = df['hadm_id'].duplicated().sum()
    if duplicate_ids > 0:
        print(f"Warning: {duplicate_ids} duplicate hadm_id values found")
    
    return valid_records

def detect_model_version(model_path):
    """Detect if model is v3.0 (with metadata) or legacy"""
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    return os.path.exists(metadata_path)

def predict_ohca(model_path, data_path, output_path=None, batch_processing=False, 
                chunk_size=10000, verbose=False):
    """
    Apply OHCA model to discharge notes
    
    Args:
        model_path: Path to trained model
        data_path: Path to CSV with discharge notes
        output_path: Where to save results (optional)
        batch_processing: Whether to use batch processing for large datasets
        chunk_size: Chunk size for batch processing
        verbose: Enable verbose output
    """
    print("OHCA Classifier Prediction")
    print("="*40)
    
    # Validate model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path must be a directory: {model_path}")
    
    # Validate data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    # Detect model version
    is_v3_model = detect_model_version(model_path)
    model_type = "v3.0 (with optimal threshold)" if is_v3_model else "Legacy (default threshold)"
    print(f"Model type: {model_type}")
    
    # Set default output path
    if output_path is None:
        base_name = Path(data_path).stem
        output_path = f"{base_name}_ohca_predictions.csv"
    
    print(f"Output: {output_path}")
    
    # Load and validate data for size estimation
    try:
        # Read just a few rows to check format
        df_sample = pd.read_csv(data_path, nrows=5)
        validate_discharge_data(df_sample)
        
        # Get total row count
        total_rows = sum(1 for line in open(data_path)) - 1  # Subtract header
        print(f"Dataset size: {total_rows:,} records")
        
    except Exception as e:
        raise RuntimeError(f"Error reading data file: {e}")
    
    # Choose processing method based on dataset size and user preference
    if batch_processing or total_rows > chunk_size:
        if total_rows > chunk_size:
            print(f"Large dataset detected ({total_rows:,} records)")
            print(f"Using batch processing with chunk size: {chunk_size:,}")
        
        # Use batch processing
        if is_v3_model:
            print("Using v3.0 batch processing with optimal threshold...")
            try:
                final_output = process_large_dataset_with_optimal_threshold(
                    model_path, data_path, output_path, chunk_size
                )
                results = pd.read_csv(final_output)
            except Exception as e:
                raise RuntimeError(f"Batch processing failed: {e}")
        else:
            print("Warning: Legacy model detected, using standard processing")
            # For legacy models, fall back to standard processing
            df = pd.read_csv(data_path)
            validate_discharge_data(df)
            results = quick_inference(model_path, df, output_path)
    else:
        # Standard processing
        print(f"Processing {total_rows:,} records...")
        df = pd.read_csv(data_path)
        validate_discharge_data(df)
        
        if is_v3_model:
            print("Using v3.0 inference with optimal threshold...")
            results = quick_inference_with_optimal_threshold(model_path, df, output_path)
        else:
            print("Using legacy inference with default threshold...")
            results = quick_inference(model_path, df, output_path)
    
    # Analyze results
    if len(results) == 0:
        raise RuntimeError("No results generated")
    
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
        # Fallback for legacy results
        ohca_detected = (results['ohca_probability'] >= threshold_used).sum()
    
    # Confidence levels
    high_confidence = (results['ohca_probability'] >= 0.8).sum()
    very_high_confidence = (results['ohca_probability'] >= 0.9).sum()
    medium_confidence = (results['ohca_probability'] >= 0.6).sum() - high_confidence
    
    # Clinical priorities (if available)
    if 'clinical_priority' in results.columns:
        priority_dist = results['clinical_priority'].value_counts()
        immediate_review = priority_dist.get('Immediate Review', 0)
        priority_review = priority_dist.get('Priority Review', 0)
    else:
        immediate_review = very_high_confidence
        priority_review = high_confidence - very_high_confidence
    
    # Results summary
    print(f"\n" + "="*40)
    print(f"PREDICTION RESULTS SUMMARY")
    print(f"="*40)
    print(f"Total cases analyzed: {len(results):,}")
    print(f"OHCA detected: {ohca_detected:,} ({ohca_detected/len(results)*100:.1f}%)")
    print(f"Threshold used: {threshold_used:.3f} ({'optimal' if is_optimal else 'default'})")
    
    print(f"\nConfidence Distribution:")
    print(f"  Very high (≥90%): {very_high_confidence:,} cases")
    print(f"  High (80-90%): {high_confidence - very_high_confidence:,} cases")
    print(f"  Medium (60-80%): {medium_confidence:,} cases")
    print(f"  Lower confidence: {len(results) - medium_confidence - high_confidence:,} cases")
    
    if 'clinical_priority' in results.columns:
        print(f"\nClinical Priority Distribution:")
        for priority, count in priority_dist.items():
            print(f"  {priority}: {count:,} cases")
    
    # Show highest probability cases
    print(f"\nTop 10 Highest Probability Cases:")
    top_cases = results.nlargest(min(10, len(results)), 'ohca_probability')
    for i, (_, row) in enumerate(top_cases.iterrows(), 1):
        prob = row['ohca_probability']
        hadm_id = row['hadm_id']
        pred = "OHCA" if 'ohca_prediction' in row and row['ohca_prediction'] == 1 else f"OHCA (>{threshold_used:.2f})" if prob >= threshold_used else "Non-OHCA"
        priority = f" [{row.get('clinical_priority', 'N/A')}]" if 'clinical_priority' in row else ""
        print(f"  {i:2d}. {hadm_id}: {prob:.3f} → {pred}{priority}")
    
    print(f"\nResults saved to: {output_path}")
    
    # Clinical recommendations
    if immediate_review > 0 or priority_review > 0:
        print(f"\n🏥 CLINICAL RECOMMENDATIONS:")
        if immediate_review > 0:
            print(f"  🔴 IMMEDIATE REVIEW: {immediate_review:,} cases (≥90% probability)")
        if priority_review > 0:
            print(f"  🟡 PRIORITY REVIEW: {priority_review:,} cases (80-90% probability)")
        if is_optimal:
            print(f"  ✅ Predictions use optimal threshold from validation data")
        else:
            print(f"  ⚠️  Legacy model: Consider retraining with v3.0 methodology")
    
    # Additional analysis if verbose
    if verbose:
        print(f"\n" + "="*40)
        print(f"DETAILED ANALYSIS")
        print(f"="*40)
        try:
            analysis = analyze_predictions_enhanced(results)
            
            if 'statistics' in analysis:
                stats = analysis['statistics']
                print(f"Statistical Summary:")
                print(f"  Mean probability: {stats.get('mean_probability', 0):.4f}")
                print(f"  Std deviation: {stats.get('std_probability', 0):.4f}")
                print(f"  Median probability: {stats.get('median_probability', 0):.4f}")
            
        except Exception as e:
            print(f"Warning: Could not generate detailed analysis: {e}")
    
    return results

def main():
    """Main function for console script entry point"""
    parser = argparse.ArgumentParser(
        description='Apply OHCA classifier to discharge notes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  ohca-predict ./my_model discharge_notes.csv
  
  # With custom output and verbose mode
  ohca-predict ./my_model data.csv --output predictions.csv --verbose
  
  # Force batch processing for large datasets
  ohca-predict ./my_model large_data.csv --batch --chunk-size 5000
  
  # Quick analysis without saving
  ohca-predict ./my_model data.csv --no-save --verbose

Input CSV should have columns:
  hadm_id: Unique admission identifier
  clean_text: Discharge note text
        """
    )
    
    parser.add_argument('model_path', help='Path to trained model directory')
    parser.add_argument('data_path', help='Path to CSV file with discharge notes')
    parser.add_argument('--output', '-o', help='Output CSV path (default: auto-generated)')
    parser.add_argument('--batch', action='store_true',
                       help='Force batch processing (useful for large datasets)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for batch processing (default: 10000)')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to file (analysis only)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output with detailed analysis')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (only results summary)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        print("Train a model first using: ohca-train your_labeled_data.csv")
        return 1
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("\nYour CSV file should have columns:")
        print("  hadm_id: Unique admission identifier")
        print("  clean_text: Discharge note text")
        print("\nPrepare your data using: ohca-prepare discharge your_raw_data.csv")
        return 1
    
    if args.chunk_size < 100:
        print(f"Error: chunk-size must be at least 100, got: {args.chunk_size}")
        return 1
    
    if args.quiet and args.verbose:
        print("Error: Cannot use both --quiet and --verbose options")
        return 1
    
    # Handle no-save option
    output_path = None if args.no_save else args.output
    
    try:
        # Redirect stdout for quiet mode
        if args.quiet:
            import io
            import contextlib
            
            # Capture stdout but still allow error messages
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            
            with contextlib.redirect_stdout(captured_output):
                results = predict_ohca(
                    args.model_path, 
                    args.data_path, 
                    output_path,
                    batch_processing=args.batch,
                    chunk_size=args.chunk_size,
                    verbose=args.verbose
                )
            
            # Print only essential summary in quiet mode
            sys.stdout = old_stdout
            ohca_detected = results['ohca_prediction'].sum() if 'ohca_prediction' in results.columns else (results['ohca_probability'] >= 0.5).sum()
            print(f"Processed {len(results):,} cases, detected {ohca_detected:,} OHCA cases")
            if not args.no_save and output_path:
                print(f"Results: {output_path}")
        else:
            results = predict_ohca(
                args.model_path, 
                args.data_path, 
                output_path,
                batch_processing=args.batch,
                chunk_size=args.chunk_size,
                verbose=args.verbose
            )
        
        # Additional verbose output
        if args.verbose and not args.quiet:
            print(f"\nFile Information:")
            print(f"  Model type: {'v3.0' if detect_model_version(args.model_path) else 'Legacy'}")
            print(f"  Input size: {os.path.getsize(args.data_path):,} bytes")
            if output_path and os.path.exists(output_path):
                print(f"  Output size: {os.path.getsize(output_path):,} bytes")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user")
        return 1
    except Exception as e:
        print(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
