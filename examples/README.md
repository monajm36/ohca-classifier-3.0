# OHCA Classifier v3.0 - Usage Examples

## Installation First

```bash
# Clone and install the package
git clone https://github.com/monajm36/ohca-classifier-3.0.git
cd ohca-classifier-3.0
pip install -e .
```

## Quick Start for Real Data

### Method 1: Using Console Scripts (Recommended)

After installation, you can use the console commands from anywhere:

```bash
# 1. Prepare your labeled training data
ohca-prepare labeled your_raw_training_data.csv

# 2. Train the model
ohca-train your_raw_training_data_prepared.csv --model_path ./my_ohca_model

# 3. Prepare discharge notes for prediction
ohca-prepare discharge your_discharge_notes.csv

# 4. Apply the trained model
ohca-predict ./my_ohca_model your_discharge_notes_prepared.csv
```

### Method 2: Direct Python Execution

If you prefer running scripts directly:

```bash
# 1. Prepare training data
python scripts/prepare_data.py labeled your_raw_training_data.csv

# 2. Train model
python scripts/train_from_labeled_data.py your_raw_training_data_prepared.csv

# 3. Prepare discharge notes
python scripts/prepare_data.py discharge your_discharge_notes.csv

# 4. Make predictions
python scripts/predict_ohca.py ./trained_ohca_model your_discharge_notes_prepared.csv
```

## Detailed Command Examples

### Data Preparation

```bash
# Interactive preparation (will prompt for column mapping)
ohca-prepare labeled training_data.csv

# Non-interactive with custom column mapping
ohca-prepare labeled data.csv --no-interactive --map admission_id:hadm_id --map note_text:clean_text

# Prepare discharge notes with custom output
ohca-prepare discharge notes.csv --output prepared_notes.csv --verbose
```

### Training Models

```bash
# Basic training
ohca-train labeled_data_prepared.csv

# Advanced training options
ohca-train data.csv --model_path ./custom_model --epochs 5 --test_size 0.15 --verbose

# Training with specific random seed for reproducibility
ohca-train data.csv --model_path ./reproducible_model --random_state 123
```

### Making Predictions

```bash
# Basic prediction
ohca-predict ./my_model discharge_notes_prepared.csv

# With custom output and verbose analysis
ohca-predict ./my_model notes.csv --output detailed_predictions.csv --verbose

# Batch processing for large datasets
ohca-predict ./my_model large_dataset.csv --batch --chunk-size 5000

# Quick analysis without saving results
ohca-predict ./my_model data.csv --no-save --verbose

# Minimal output for automation/scripting
ohca-predict ./my_model data.csv --quiet
```

## Data Format Requirements

### For Training (Labeled Data)

Your CSV must have these columns:
- `hadm_id`: Unique admission identifier (string/int)
- `clean_text`: Discharge note text (string)
- `ohca_label`: OHCA label - 1 for OHCA, 0 for non-OHCA (int)
- `subject_id`: Patient identifier (optional - will use hadm_id if missing)
- `confidence`: Annotation confidence 1-5 (optional - defaults to 4)

**Example:**
```csv
hadm_id,subject_id,clean_text,ohca_label,confidence
12345,101,"Patient found down at home, CPR initiated...",1,5
12346,102,"Chest pain, no arrest, normal workup...",0,4
```

### For Prediction (Discharge Notes)

Your CSV must have these columns:
- `hadm_id`: Unique admission identifier (string/int)
- `clean_text`: Discharge note text (string)

**Example:**
```csv
hadm_id,clean_text
67890,"Chief complaint: Cardiac arrest. Patient collapsed..."
67891,"Chief complaint: Chest pain. Patient stable..."
```

## Expected Output Formats

### Training Output

After training, you'll get:
- Model directory with trained model files
- `model_metadata.json` with optimal threshold and performance metrics
- Console output showing validation performance

### Prediction Output

Prediction results include:
- `hadm_id`: Original admission ID
- `ohca_probability`: OHCA probability (0-1)
- `ohca_prediction`: Binary prediction using optimal threshold
- `clinical_priority`: Clinical priority level (v3.0 models)
- `confidence_category`: Confidence category (v3.0 models)
- `optimal_threshold_used`: Threshold used for predictions

## Advanced Usage Examples

### Cross-Institutional Deployment

```bash
# Train on your institution's data
ohca-train your_institution_data.csv --model_path ./institution_model

# Apply to external dataset (like CLIF)
ohca-predict ./institution_model clif_discharge_notes.csv --output clif_predictions.csv --verbose
```

### Model Comparison

```bash
# Train multiple models with different parameters
ohca-train data.csv --model_path ./model_3epochs --epochs 3
ohca-train data.csv --model_path ./model_5epochs --epochs 5

# Compare predictions
ohca-predict ./model_3epochs test_data.csv --output pred_3epochs.csv
ohca-predict ./model_5epochs test_data.csv --output pred_5epochs.csv
```

### Batch Processing Workflow

```bash
# For very large datasets (>100K records)
ohca-predict ./my_model huge_dataset.csv --batch --chunk-size 10000 --output results.csv

# Monitor progress with verbose output
ohca-predict ./my_model large_data.csv --batch --verbose
```

## Integration with Python Code

### Using the Package in Your Scripts

```python
from src import quick_inference_with_optimal_threshold, complete_improved_training_pipeline

# Training
result = complete_improved_training_pipeline(
    data_path="discharge_notes.csv",
    annotation_dir="./annotation_v3"
)

# After manual annotation...
# (Use the training functions)

# Inference
predictions = quick_inference_with_optimal_threshold(
    model_path="./trained_model",
    data_path="new_discharge_notes.csv"
)
```

### Working with Results

```python
import pandas as pd

# Load prediction results
results = pd.read_csv("predictions.csv")

# Get high-priority cases
immediate_review = results[results['clinical_priority'] == 'Immediate Review']
priority_review = results[results['clinical_priority'] == 'Priority Review']

print(f"Cases needing immediate review: {len(immediate_review)}")
print(f"Cases needing priority review: {len(priority_review)}")
```

## Troubleshooting Common Issues

### Data Preparation Issues

```bash
# If column names don't match requirements
ohca-prepare labeled data.csv --map your_id_column:hadm_id --map your_text_column:clean_text

# Check your data format
head -5 your_data.csv  # Verify columns and format
```

### Training Issues

```bash
# If you get memory errors, reduce batch size internally by training on smaller subset first
# If convergence issues, try more epochs
ohca-train data.csv --epochs 5

# For reproducibility issues
ohca-train data.csv --random_state 42
```

### Prediction Issues

```bash
# For large datasets causing memory issues
ohca-predict ./model data.csv --batch --chunk-size 1000

# For debugging
ohca-predict ./model data.csv --verbose
```

## Getting Help

```bash
# Get help for any command
ohca-prepare --help
ohca-train --help
ohca-predict --help

# Check package installation
python -c "import src; print(src.get_version())"
```

## Demo Examples

Run the included demo examples in the `examples/` directory:

```bash
cd examples
python training_example.py    # Shows v3.0 methodology
python inference_example.py  # Shows inference workflows
python clif_dataset_example.py  # Cross-institutional deployment
```

## Model Versions

The package supports both model types:
- **v3.0 models**: Use optimal thresholds and enhanced clinical decision support
- **Legacy models**: Fallback to default threshold (0.5) with compatibility warnings

Console scripts automatically detect model version and use appropriate methods.
