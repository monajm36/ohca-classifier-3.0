# OHCA Inference Module v3.0 - Improved with Optimal Threshold Support
# Apply pre-trained OHCA classifier to new datasets using optimal thresholds

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference Module v3.0 - Using device: {DEVICE}")

# =============================================================================
# INFERENCE DATASET CLASS
# =============================================================================

class OHCAInferenceDataset(Dataset):
    """Dataset for OHCA inference on new data"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        if 'hadm_id' not in self.data.columns or 'clean_text' not in self.data.columns:
            raise ValueError("DataFrame must contain 'hadm_id' and 'clean_text' columns")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_text'])
        
        # Apply preprocessing consistent with training
        if 'transfer' in text.lower():
            text = "TRANSFERRED_PATIENT " + text
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hadm_id': row['hadm_id']
        }

# =============================================================================
# IMPROVED MODEL LOADING FUNCTIONS
# =============================================================================

def load_ohca_model_with_metadata(model_path):
    """
    Load pre-trained OHCA model, tokenizer, and metadata (including optimal threshold).
    This addresses the data scientist's feedback about using consistent thresholds.
    
    Args:
        model_path: Path to saved model directory
    
    Returns:
        tuple: (model, tokenizer, optimal_threshold, metadata)
    """
    print(f"Loading OHCA model with metadata from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(DEVICE)
        model.eval()
        
        # Load metadata with optimal threshold
        metadata_path = os.path.join(model_path, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            optimal_threshold = metadata.get('optimal_threshold', 0.5)
            print(f"Loaded optimal threshold: {optimal_threshold:.3f}")
            print(f"Model version: {metadata.get('model_version', 'unknown')}")
        else:
            print("Warning: No metadata file found. Using default threshold of 0.5")
            optimal_threshold = 0.5
            metadata = {'optimal_threshold': 0.5, 'model_version': 'legacy'}
        
        print("Model loaded successfully")
        print(f"   Device: {DEVICE}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Optimal threshold: {optimal_threshold:.3f}")
        
        return model, tokenizer, optimal_threshold, metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def load_ohca_model(model_path):
    """
    Backward compatibility function - loads model without metadata
    
    Args:
        model_path: Path to saved model directory
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading OHCA model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(DEVICE)
        model.eval()
        
        print("Model loaded successfully (legacy mode)")
        print(f"   Device: {DEVICE}")
        print(f"   Model type: {type(model).__name__}")
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# =============================================================================
# IMPROVED INFERENCE FUNCTIONS
# =============================================================================

def categorize_confidence_with_optimal_threshold(prob, optimal_threshold):
    """
    Categorize confidence levels relative to optimal threshold
    
    Args:
        prob: Probability score
        optimal_threshold: Optimal threshold from training
        
    Returns:
        tuple: (confidence_category, clinical_priority)
    """
    if prob >= 0.9:
        return "Very High", "Immediate Review"
    elif prob >= 0.7:
        return "High", "Priority Review"  
    elif prob >= optimal_threshold:
        return "Medium-High", "Clinical Review"
    elif prob >= 0.3:
        return "Medium", "Consider Review"
    elif prob >= 0.1:
        return "Low", "Routine Processing"
    else:
        return "Very Low", "Routine Processing"

def run_inference_with_optimal_threshold(model, tokenizer, inference_df, 
                                       optimal_threshold=0.5, batch_size=16, 
                                       output_path=None):
    """
    Run OHCA inference using the optimal threshold from training.
    This addresses the data scientist's feedback about threshold consistency.
    
    Args:
        model: Pre-trained OHCA model
        tokenizer: Model tokenizer
        inference_df: DataFrame with columns ['hadm_id', 'clean_text']
        optimal_threshold: Optimal threshold from validation set
        batch_size: Batch size for inference
        output_path: Optional path to save results CSV
    
    Returns:
        DataFrame: Results with probabilities and predictions using optimal threshold
    """
    print(f"Running OHCA inference on {len(inference_df):,} cases...")
    print(f"Using optimal threshold: {optimal_threshold:.3f}")
    
    # Validate input data
    required_cols = ['hadm_id', 'clean_text']
    missing_cols = [col for col in required_cols if col not in inference_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any rows with missing data
    clean_df = inference_df.dropna(subset=required_cols).copy()
    if len(clean_df) < len(inference_df):
        print(f"Warning: Removed {len(inference_df) - len(clean_df)} rows with missing data")
    
    # Create dataset and dataloader
    inference_dataset = OHCAInferenceDataset(clean_df, tokenizer)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    # Run inference
    model.eval()
    all_probabilities = []
    all_hadm_ids = []
    
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Processing batches"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            hadm_ids = batch['hadm_id']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Get OHCA probabilities (class 1)
            ohca_probs = probs[:, 1].cpu().numpy()
            
            all_probabilities.extend(ohca_probs)
            all_hadm_ids.extend(hadm_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'hadm_id': all_hadm_ids,
        'ohca_probability': all_probabilities
    })
    
    # Add prediction using optimal threshold (primary prediction)
    results_df['ohca_prediction'] = (results_df['ohca_probability'] >= optimal_threshold).astype(int)
    results_df['optimal_threshold_used'] = optimal_threshold
    
    # Add legacy predictions for comparison
    results_df['prediction_050'] = (results_df['ohca_probability'] >= 0.5).astype(int)
    results_df['prediction_070'] = (results_df['ohca_probability'] >= 0.7).astype(int)
    results_df['prediction_090'] = (results_df['ohca_probability'] >= 0.9).astype(int)
    
    # Add improved confidence categories and clinical priorities
    confidence_info = [categorize_confidence_with_optimal_threshold(prob, optimal_threshold) 
                      for prob in results_df['ohca_probability']]
    results_df['confidence_category'] = [info[0] for info in confidence_info]
    results_df['clinical_priority'] = [info[1] for info in confidence_info]
    
    # Add interpretation column
    results_df['interpretation'] = results_df.apply(
        lambda row: f"OHCA detected (p={row['ohca_probability']:.3f})" 
        if row['ohca_prediction'] == 1 
        else f"No OHCA (p={row['ohca_probability']:.3f})", axis=1
    )
    
    # Sort by probability (highest first)
    results_df = results_df.sort_values('ohca_probability', ascending=False).reset_index(drop=True)
    
    # Print improved summary
    print(f"\nInference Results Summary:")
    print(f"   Total cases processed: {len(results_df):,}")
    print(f"   Mean OHCA probability: {results_df['ohca_probability'].mean():.4f}")
    print(f"   OHCA detected (optimal threshold): {results_df['ohca_prediction'].sum():,}")
    print(f"   Detection rate: {results_df['ohca_prediction'].mean()*100:.2f}%")
    
    # Clinical priority distribution
    print(f"\nClinical Priority Distribution:")
    priority_dist = results_df['clinical_priority'].value_counts()
    for priority, count in priority_dist.items():
        pct = count / len(results_df) * 100
        print(f"   {priority}: {count:,} cases ({pct:.1f}%)")
    
    # Confidence distribution
    print(f"\nConfidence Distribution:")
    conf_dist = results_df['confidence_category'].value_counts()
    for category, count in conf_dist.items():
        pct = count / len(results_df) * 100
        print(f"   {category}: {count:,} cases ({pct:.1f}%)")
    
    # Save results if path provided
    if output_path:
        # Add metadata to the saved file
        results_df['inference_date'] = pd.Timestamp.now().isoformat()
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results_df

def run_inference(model, tokenizer, inference_df, batch_size=16, 
                 output_path=None, probability_threshold=0.5):
    """
    Legacy inference function for backward compatibility
    """
    print("Warning: Using legacy inference function. Consider upgrading to run_inference_with_optimal_threshold()")
    return run_inference_with_optimal_threshold(
        model, tokenizer, inference_df, probability_threshold, batch_size, output_path
    )

# =============================================================================
# IMPROVED CONVENIENCE FUNCTIONS
# =============================================================================

def quick_inference_with_optimal_threshold(model_path, data_path, output_path=None):
    """
    Quick inference function that automatically uses the optimal threshold.
    This is the recommended way to run inference with v3.0 models.
    
    Args:
        model_path: Path to trained model (must include metadata)
        data_path: Path to input CSV (or DataFrame)
        output_path: Optional output path
    
    Returns:
        DataFrame: Inference results using optimal threshold
    """
    print("Quick OHCA Inference v3.0 with Optimal Threshold")
    
    # Load model with metadata
    model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
    
    # Load data
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} cases from {data_path}")
    else:
        df = data_path.copy()
        print(f"Processing {len(df):,} cases from DataFrame")
    
    # Run inference with optimal threshold
    results = run_inference_with_optimal_threshold(
        model, tokenizer, df, optimal_threshold, output_path=output_path
    )
    
    # Enhanced summary
    ohca_cases = results['ohca_prediction'].sum()
    high_priority = (results['clinical_priority'] == 'Immediate Review').sum()
    priority = (results['clinical_priority'] == 'Priority Review').sum()
    
    print(f"\nEnhanced Summary:")
    print(f"   OHCA detected (optimal threshold): {ohca_cases:,}")
    print(f"   Immediate review needed: {high_priority:,}")
    print(f"   Priority review needed: {priority:,}")
    print(f"   Model version: {metadata.get('model_version', 'unknown')}")
    print(f"   Optimal threshold used: {optimal_threshold:.3f}")
    
    return results

def quick_inference(model_path, data_path, output_path=None):
    """
    Backward compatible quick inference function.
    Automatically detects if model has metadata and uses optimal threshold if available.
    
    Args:
        model_path: Path to trained model
        data_path: Path to input CSV (or DataFrame)  
        output_path: Optional output path
    
    Returns:
        DataFrame: Inference results
    """
    print("Quick OHCA Inference")
    
    # Try to load with metadata first
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_path):
        print("Detected v3.0 model with metadata - using optimal threshold")
        return quick_inference_with_optimal_threshold(model_path, data_path, output_path)
    else:
        print("Detected legacy model - using default threshold 0.5")
        # Load model without metadata
        model, tokenizer = load_ohca_model(model_path)
        
        # Load data
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df):,} cases from {data_path}")
        else:
            df = data_path.copy()
            print(f"Processing {len(df):,} cases from DataFrame")
        
        # Run inference with default threshold
        results = run_inference_with_optimal_threshold(
            model, tokenizer, df, optimal_threshold=0.5, output_path=output_path
        )
        
        # Quick summary
        ohca_cases = results['ohca_prediction'].sum()
        high_conf = (results['ohca_probability'] >= 0.8).sum()
        
        print(f"\nQuick Summary:")
        print(f"   Predicted OHCA cases: {ohca_cases:,}")
        print(f"   High confidence: {high_conf:,}")
        
        return results

def analyze_predictions_enhanced(results_df):
    """
    Enhanced prediction analysis with optimal threshold insights
    
    Args:
        results_df: Results from inference with optimal threshold
    
    Returns:
        dict: Enhanced analysis summary
    """
    print("Analyzing prediction patterns with optimal threshold insights...")
    
    optimal_threshold = results_df['optimal_threshold_used'].iloc[0] if 'optimal_threshold_used' in results_df.columns else 0.5
    
    # Basic statistics
    stats = {
        'total_cases': len(results_df),
        'optimal_threshold_used': optimal_threshold,
        'mean_probability': results_df['ohca_probability'].mean(),
        'std_probability': results_df['ohca_probability'].std(),
        'median_probability': results_df['ohca_probability'].median(),
        'ohca_detected_optimal': results_df.get('ohca_prediction', []).sum(),
        'high_confidence_cases': (results_df['ohca_probability'] >= 0.8).sum(),
        'predicted_ohca_050': results_df.get('prediction_050', []).sum(),
        'predicted_ohca_070': results_df.get('prediction_070', []).sum(),
        'predicted_ohca_090': results_df.get('prediction_090', []).sum(),
    }
    
    # Clinical priority distribution
    if 'clinical_priority' in results_df.columns:
        priority_dist = results_df['clinical_priority'].value_counts().to_dict()
    else:
        priority_dist = {}
    
    # Confidence distribution  
    if 'confidence_category' in results_df.columns:
        conf_dist = results_df['confidence_category'].value_counts().to_dict()
    else:
        conf_dist = {}
    
    # Print enhanced analysis
    print(f"\nEnhanced Prediction Analysis:")
    print(f"   Total cases: {stats['total_cases']:,}")
    print(f"   Optimal threshold used: {stats['optimal_threshold_used']:.3f}")
    print(f"   Mean probability: {stats['mean_probability']:.4f}")
    print(f"   OHCA detected (optimal): {stats['ohca_detected_optimal']:,}")
    
    if stats['ohca_detected_optimal'] > 0:
        prevalence = stats['ohca_detected_optimal'] / stats['total_cases'] * 100
        print(f"   Estimated OHCA prevalence: {prevalence:.2f}%")
    
    # Comparison with static thresholds
    print(f"\nThreshold Comparison:")
    print(f"   Optimal threshold ({optimal_threshold:.3f}): {stats['ohca_detected_optimal']:,} cases")
    print(f"   Static threshold (0.5): {stats['predicted_ohca_050']:,} cases")
    print(f"   Static threshold (0.7): {stats['predicted_ohca_070']:,} cases")
    
    # Clinical recommendations
    print(f"\nClinical Recommendations:")
    if priority_dist:
        for priority, count in priority_dist.items():
            if count > 0:
                print(f"   {priority}: {count:,} cases")
    
    return {
        'statistics': stats,
        'clinical_priority_distribution': priority_dist,
        'confidence_distribution': conf_dist,
        'optimal_threshold': optimal_threshold,
        'high_confidence_cases': results_df[results_df['ohca_probability'] >= 0.8] if len(results_df) > 0 else pd.DataFrame()
    }

# =============================================================================
# ENHANCED BATCH PROCESSING
# =============================================================================

def process_large_dataset_with_optimal_threshold(model_path, data_path, output_path, 
                                                chunk_size=10000, batch_size=16):
    """
    Process large datasets using optimal threshold from model metadata
    
    Args:
        model_path: Path to trained model with metadata
        data_path: Path to input CSV file
        output_path: Path for output results
        chunk_size: Number of rows per chunk
        batch_size: Batch size for inference
    
    Returns:
        str: Path to completed results file
    """
    print(f"Processing large dataset in chunks of {chunk_size:,} with optimal threshold...")
    
    # Load model with metadata once
    model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
    
    # Read data in chunks
    chunk_results = []
    chunk_num = 0
    
    for chunk_df in pd.read_csv(data_path, chunksize=chunk_size):
        chunk_num += 1
        print(f"\nProcessing chunk {chunk_num} ({len(chunk_df):,} rows)...")
        
        # Run inference on chunk with optimal threshold
        chunk_result = run_inference_with_optimal_threshold(
            model, tokenizer, chunk_df, optimal_threshold,
            batch_size=batch_size, output_path=None
        )
        
        chunk_results.append(chunk_result)
        
        # Save intermediate results
        temp_path = f"{output_path}.chunk_{chunk_num}.csv"
        chunk_result.to_csv(temp_path, index=False)
        print(f"Chunk {chunk_num} saved to: {temp_path}")
    
    # Combine all chunks
    print(f"\nCombining {len(chunk_results)} chunks...")
    final_results = pd.concat(chunk_results, ignore_index=True)
    
    # Sort by probability and save
    final_results = final_results.sort_values('ohca_probability', ascending=False)
    
    # Add final metadata
    final_results['model_version'] = metadata.get('model_version', 'unknown')
    final_results['processing_date'] = pd.Timestamp.now().isoformat()
    
    final_results.to_csv(output_path, index=False)
    
    print(f"Complete results saved to: {output_path}")
    print(f"Total cases processed: {len(final_results):,}")
    print(f"OHCA detected with optimal threshold: {final_results['ohca_prediction'].sum():,}")
    
    # Clean up intermediate files
    for i in range(1, chunk_num + 1):
        temp_path = f"{output_path}.chunk_{i}.csv"
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return output_path

# Legacy batch processing function
def process_large_dataset(model_path, data_path, output_path, 
                         chunk_size=10000, batch_size=16):
    """Legacy function for backward compatibility"""
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_path):
        return process_large_dataset_with_optimal_threshold(
            model_path, data_path, output_path, chunk_size, batch_size
        )
    else:
        print("Warning: Legacy model detected. Using default threshold processing.")
        # Fall back to original implementation
        model, tokenizer = load_ohca_model(model_path)
        
        chunk_results = []
        chunk_num = 0
        
        for chunk_df in pd.read_csv(data_path, chunksize=chunk_size):
            chunk_num += 1
            print(f"\nProcessing chunk {chunk_num} ({len(chunk_df):,} rows)...")
            
            chunk_result = run_inference_with_optimal_threshold(
                model, tokenizer, chunk_df, optimal_threshold=0.5,
                batch_size=batch_size, output_path=None
            )
            
            chunk_results.append(chunk_result)
        
        final_results = pd.concat(chunk_results, ignore_index=True)
        final_results = final_results.sort_values('ohca_probability', ascending=False)
        final_results.to_csv(output_path, index=False)
        
        return output_path

# =============================================================================
# ENHANCED TESTING FUNCTIONS
# =============================================================================

def test_model_on_sample(model_path, sample_texts):
    """
    Test model on sample texts using optimal threshold if available
    
    Args:
        model_path: Path to trained model
        sample_texts: List of text strings or dict with hadm_id: text
    
    Returns:
        DataFrame: Test results with optimal threshold predictions
    """
    print("Testing model on sample texts...")
    
    # Prepare test data
    if isinstance(sample_texts, dict):
        test_df = pd.DataFrame([
            {'hadm_id': hadm_id, 'clean_text': text} 
            for hadm_id, text in sample_texts.items()
        ])
    else:
        test_df = pd.DataFrame([
            {'hadm_id': f'TEST_{i:03d}', 'clean_text': text} 
            for i, text in enumerate(sample_texts, 1)
        ])
    
    # Try to load with metadata
    metadata_path = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_path):
        model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata(model_path)
        results = run_inference_with_optimal_threshold(
            model, tokenizer, test_df, optimal_threshold, output_path=None
        )
    else:
        model, tokenizer = load_ohca_model(model_path)
        results = run_inference_with_optimal_threshold(
            model, tokenizer, test_df, optimal_threshold=0.5, output_path=None
        )
    
    # Print enhanced results
    print(f"\nTest Results:")
    for _, row in results.iterrows():
        prob = row['ohca_probability']
        pred = "OHCA" if row['ohca_prediction'] == 1 else "Non-OHCA"
        conf = row['confidence_category']
        priority = row['clinical_priority']
        
        print(f"   {row['hadm_id']}: {pred} (p={prob:.3f}, {conf}, {priority})")
        
        # Show text preview
        text_preview = test_df[test_df['hadm_id']==row['hadm_id']]['clean_text'].iloc[0]
        print(f"      Text: {text_preview[:100]}...")
        print()
    
    return results

# =============================================================================
# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

def get_high_confidence_cases(results_df, threshold=0.8, max_cases=100):
    """Extract high-confidence OHCA predictions for manual review"""
    high_conf = results_df[results_df['ohca_probability'] >= threshold].copy()
    high_conf = high_conf.head(max_cases)
    
    print(f"Found {len(high_conf)} high-confidence cases (≥{threshold})")
    
    return high_conf

def analyze_predictions(results_df, original_df=None):
    """Legacy analysis function - redirects to enhanced version"""
    return analyze_predictions_enhanced(results_df)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("OHCA Inference Module v3.0 - Enhanced with Optimal Threshold Support")
    print("="*75)
    print("Key improvements:")
    print("✅ Automatic optimal threshold loading and usage")
    print("✅ Enhanced confidence categories based on optimal threshold")  
    print("✅ Clinical priority recommendations")
    print("✅ Backward compatibility with legacy models")
    print("✅ Enhanced analysis and reporting")
    print()
    print("Main functions:")
    print("• quick_inference_with_optimal_threshold() - Recommended for v3.0 models")
    print("• load_ohca_model_with_metadata() - Load model with optimal threshold")
    print("• run_inference_with_optimal_threshold() - Enhanced inference")
    print("• process_large_dataset_with_optimal_threshold() - Batch processing")
    print("• analyze_predictions_enhanced() - Enhanced prediction analysis")
    print()
    print("Legacy functions (maintained for compatibility):")
    print("• quick_inference() - Auto-detects model version")
    print("• load_ohca_model() - Basic model loading")
    print("• run_inference() - Basic inference")
    print()
    print("See examples/ folder for detailed usage examples.")
    print("="*75)
