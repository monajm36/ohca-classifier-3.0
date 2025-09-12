# OHCA Classifier v3.0 - Improved Methodology
BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases in medical text with enhanced machine learning methodology

## NLP OHCA Classifier v3.0
A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases in medical discharge notes using improved natural language processing methodology that addresses key methodological concerns in medical AI.

## Key Improvements in v3.0

This version implements significant methodological improvements based on data science best practices:

**Patient-Level Data Splits** - Prevents data leakage by ensuring all notes from the same patient stay in one split  
**Proper Train/Validation/Test** - Uses independent test set for unbiased evaluation  
**Optimal Threshold Finding** - Finds and saves optimal decision threshold during training  
**Larger Training Samples** - 800+ training samples instead of 264  
**Enhanced Clinical Decision Support** - Improved confidence categories and workflow integration  
**Unbiased Evaluation** - Eliminates threshold tuning on test data  

## Overview
This package provides two main modules with v3.0 enhancements:

- **Training Pipeline** (`ohca_training_pipeline.py`) - Complete workflow with improved methodology
- **Inference Module** (`ohca_inference.py`) - Apply models with optimal threshold support

## Features

### Training Pipeline (Enhanced v3.0)
- **Patient-Level Splits**: Prevents data leakage between training and test sets
- **Dual Annotation Strategy**: Separate training and validation annotation files
- **Intelligent Sampling**: Two-stage sampling strategy (keyword-enriched + random)  
- **Larger Sample Sizes**: 800 training + 200 validation samples
- **BERT-based Training**: Uses PubMedBERT optimized for medical text
- **Optimal Threshold Finding**: Automatically finds best decision threshold
- **Unbiased Evaluation**: Independent test set for reliable performance estimates

### Inference Module (Enhanced v3.0)
- **Optimal Threshold Usage**: Automatically uses threshold found during training
- **Enhanced Clinical Priorities**: Improved confidence categories for clinical workflow
- **Batch Processing**: Efficient inference on large datasets
- **Clinical Decision Support**: Evidence-based probability thresholds
- **Backward Compatibility**: Works with both v3.0 and legacy models

## Quick Start for Real Data

### Step 1: Train on Your Labeled Data
```bash
# Prepare your data (if needed)
python scripts/prepare_data.py labeled your_labeled_data.csv

# Train the model
python scripts/train_from_labeled_data.py your_labeled_data_prepared.csv
```

### Step 2: Apply to New Discharge Notes
```bash
# Prepare discharge notes (if needed)  
python scripts/prepare_data.py discharge your_discharge_notes.csv

# Apply model
python scripts/predict_ohca.py ./trained_ohca_model your_discharge_notes_prepared.csv
```

See `scripts/` folder for ready-to-use workflows and `examples/` for detailed demonstrations.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Install from source

1. Clone the repository:
```bash
git clone https://github.com/monajm36/ohca-classifier-3.0.git
cd ohca-classifier-3.0
```

2. Set up virtual environment:
```bash
python3 -m venv .venv/
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

**Note for Windows users**: Replace `source .venv/bin/activate` with `.venv\Scripts\activate`

## Training a New Model (v3.0 Methodology - RECOMMENDED)

```python
from src.ohca_training_pipeline import complete_improved_training_pipeline
import pandas as pd

# Step 1: Create patient-level splits and annotation samples
results = complete_improved_training_pipeline(
    data_path="your_discharge_notes.csv",  # Must have: hadm_id, subject_id, clean_text
    annotation_dir="./annotation_v3",
    train_sample_size=800,    # Much larger than legacy
    val_sample_size=200       # Separate validation sample
)

# Step 2: Manually annotate BOTH Excel files:
# - annotation_v3/train_annotation.xlsx (800 cases)
# - annotation_v3/validation_annotation.xlsx (200 cases)
# Label each case: 1=OHCA, 0=Non-OHCA

# Step 3: Complete training (after annotation)
from src.ohca_training_pipeline import complete_annotation_and_train_v3

model_results = complete_annotation_and_train_v3(
    train_annotation_file="./annotation_v3/train_annotation.xlsx",
    val_annotation_file="./annotation_v3/validation_annotation.xlsx",
    test_file="./annotation_v3/test_set_DO_NOT_ANNOTATE.csv",
    model_save_path="./my_ohca_model_v3",
    num_epochs=3
)

print(f"Optimal threshold: {model_results['optimal_threshold']:.3f}")
print(f"Model automatically uses this threshold during inference")
```

### Using a Pre-trained v3.0 Model

```python
from src.ohca_inference import quick_inference_with_optimal_threshold
import pandas as pd

# Apply v3.0 model to new data (uses optimal threshold automatically)
new_data = pd.read_csv("new_discharge_notes.csv")  # Must have: hadm_id, clean_text
results = quick_inference_with_optimal_threshold(
    model_path="./my_ohca_model_v3",  # v3.0 model with metadata
    data_path=new_data,
    output_path="ohca_predictions.csv"
)

# Enhanced v3.0 results with clinical priorities
immediate_review = results[results['clinical_priority'] == 'Immediate Review']
priority_review = results[results['clinical_priority'] == 'Priority Review']

print(f"Immediate review needed: {len(immediate_review)} cases")
print(f"Priority review needed: {len(priority_review)} cases")
print(f"Optimal threshold used: {results['optimal_threshold_used'].iloc[0]:.3f}")
```

### Backward Compatibility (Legacy Models)

```python
from src.ohca_inference import quick_inference

# Works with both v3.0 and legacy models
results = quick_inference(
    model_path="./any_model",  # Auto-detects model version
    data_path="new_data.csv"
)
```

## Data Format

### Input Requirements (Enhanced for v3.0)
Your CSV file must contain:
- `hadm_id`: Unique identifier for each hospital admission
- `subject_id`: Patient identifier (for patient-level splits to prevent data leakage)
- `clean_text`: Preprocessed discharge note text

**Example:**
```csv
hadm_id,subject_id,clean_text
12345,101,"Chief complaint: Cardiac arrest at home. Patient found down by family..."
12346,102,"Chief complaint: Chest pain. Patient presents with acute onset chest pain..."
12347,101,"Follow-up visit. Patient doing well after recent arrest..."
```

**If you don't have patient IDs**: Add this line to your preprocessing:
```python
df['subject_id'] = df['hadm_id']  # Use admission ID as patient ID
```

### Annotation Labels
- `1`: OHCA case (cardiac arrest outside hospital, primary reason for admission)
- `0`: Non-OHCA case (everything else, including transfers and historical arrests)

## Module Documentation

### Training Pipeline (Enhanced v3.0)

**Main v3.0 Functions (RECOMMENDED):**
- `complete_improved_training_pipeline()` - Create patient-level splits and annotation samples
- `complete_annotation_and_train_v3()` - Train with optimal threshold finding
- `create_patient_level_splits()` - Create proper data splits
- `find_optimal_threshold()` - Find optimal decision threshold
- `evaluate_on_test_set()` - Unbiased final evaluation

**Legacy Functions (Backward Compatible):**
- `create_training_sample()` - Legacy single-file annotation
- `complete_annotation_and_train()` - Legacy training workflow

**Example Usage (v3.0):**
```python
from src.ohca_training_pipeline import complete_improved_training_pipeline

# Enhanced training with proper methodology
result = complete_improved_training_pipeline(
    data_path="discharge_notes.csv",
    annotation_dir="./annotation_v3",
    train_sample_size=800,
    val_sample_size=200
)
```

### Inference Module (Enhanced v3.0)

**Main v3.0 Functions (RECOMMENDED):**
- `quick_inference_with_optimal_threshold()` - Uses optimal threshold automatically
- `load_ohca_model_with_metadata()` - Load model with optimal threshold
- `run_inference_with_optimal_threshold()` - Enhanced inference
- `analyze_predictions_enhanced()` - Improved prediction analysis

**Legacy Functions (Backward Compatible):**
- `quick_inference()` - Auto-detects model version
- `load_ohca_model()` - Basic model loading
- `run_inference()` - Basic inference

**Example Usage (v3.0):**
```python
from src.ohca_inference import load_ohca_model_with_metadata, run_inference_with_optimal_threshold

# Load v3.0 model with optimal threshold
model, tokenizer, optimal_threshold, metadata = load_ohca_model_with_metadata("./trained_model")

# Run inference with optimal threshold
results = run_inference_with_optimal_threshold(model, tokenizer, new_data_df, optimal_threshold)
```

## Model Architecture
- **Base Model**: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- **Task**: Binary classification (OHCA vs Non-OHCA)
- **Max Sequence Length**: 512 tokens
- **Optimization**: AdamW with linear learning rate scheduling
- **Class Balancing**: Weighted loss + minority class oversampling
- **Threshold Selection**: Optimal threshold found via validation set (v3.0)

## Performance Metrics

### v3.0 Enhanced Evaluation
The model provides unbiased performance estimates using:
- **Independent test set** for final evaluation
- **Optimal threshold** found on validation set only
- **Patient-level splits** preventing data leakage

**Clinical Metrics:**
- **Sensitivity (Recall)**: Percentage of OHCA cases correctly identified
- **Specificity**: Percentage of non-OHCA cases correctly identified
- **Precision (PPV)**: When model predicts OHCA, percentage that are correct
- **NPV**: When model predicts non-OHCA, percentage that are correct
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Clinical Usage

### Enhanced v3.0 Clinical Decision Support

**Clinical Priorities (v3.0):**
- **Immediate Review**: Very high probability cases requiring urgent attention
- **Priority Review**: High probability cases for clinical team review
- **Clinical Review**: Medium-high probability cases above optimal threshold
- **Consider Review**: Medium probability cases for potential review
- **Routine Processing**: Low probability cases

**Optimal Threshold Usage:**
- Model automatically uses threshold found during validation
- Consistent decision-making across all datasets
- Better performance than static thresholds

**Workflow Integration:**
1. Run inference on new discharge notes (uses optimal threshold)
2. Prioritize "Immediate Review" cases for urgent manual review
3. Schedule "Priority Review" cases for clinical team evaluation
4. Use "Clinical Review" cases for quality improvement
5. Monitor routine cases for false negatives

## Repository Structure
```
ohca-classifier-3.0/
├── src/
│   ├── __init__.py
│   ├── ohca_training_pipeline.py    # Enhanced v3.0 training workflow
│   └── ohca_inference.py            # Enhanced v3.0 inference
├── scripts/
│   ├── train_from_labeled_data.py   # User-friendly training script
│   ├── predict_ohca.py              # User-friendly prediction script
│   └── prepare_data.py              # Data preparation helper
├── examples/
│   ├── training_example.py          # v3.0 training examples
│   ├── inference_example.py         # v3.0 inference examples
│   └── clif_dataset_example.py      # Cross-institutional deployment
├── docs/
│   └── annotation_guidelines.md     # Enhanced annotation guidelines
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Examples

### Complete v3.0 Training Example
```bash
cd examples
python training_example.py
# Choose option 1: v3.0 Training with Improved Methodology
```

### Enhanced v3.0 Inference Examples
```bash
cd examples
python inference_example.py
# Choose option 1: v3.0 Inference with Optimal Threshold
```

### Cross-Institutional Deployment
```bash
cd examples
python clif_dataset_example.py
# Apply v3.0 model to external datasets
```

## Advanced Usage

### Large Dataset Processing (v3.0)
```python
from src.ohca_inference import process_large_dataset_with_optimal_threshold

# Process with optimal threshold automatically
process_large_dataset_with_optimal_threshold(
    model_path="./trained_model_v3",
    data_path="large_dataset.csv",
    output_path="results.csv",
    chunk_size=5000
)
```

### Model Testing with v3.0 Features
```python
from src.ohca_inference import test_model_on_sample

# Test with optimal threshold support
test_cases = {
    'case1': "Chief complaint: Cardiac arrest at home...",
    'case2': "Chief complaint: Chest pain, no arrest..."
}

results = test_model_on_sample("./trained_model_v3", test_cases)
# Results include optimal threshold predictions and clinical priorities
```

## Performance Benchmarks

### v3.0 Methodology Performance
Typical performance with improved methodology:
- **AUC-ROC**: 0.85-0.95 (unbiased estimates)
- **Sensitivity**: 85-95% (at optimal threshold)
- **Specificity**: 85-95% (at optimal threshold)
- **F1-Score**: 0.7-0.9 (optimized via validation)

**Key Improvements over Legacy:**
- **Unbiased evaluation** using independent test set
- **Optimal threshold** provides better sensitivity/specificity balance
- **Larger training sets** (800 vs 264) improve generalization
- **Patient-level splits** prevent overoptimistic performance estimates

*Performance varies based on data quality and annotation consistency*

## Migration from Legacy Versions

### Upgrading from Legacy to v3.0

**Benefits of Upgrading:**
- More reliable performance estimates
- Better clinical decision support
- Optimal threshold usage
- Enhanced workflow integration

**Migration Steps:**
1. **Retrain with v3.0 methodology** using `complete_improved_training_pipeline()`
2. **Add patient IDs** to your data (`subject_id` column)
3. **Use v3.0 inference functions** for new predictions
4. **Update workflows** to use clinical priorities

**Backward Compatibility:**
- Legacy models continue to work
- Legacy functions automatically detect model version
- Gradual migration supported

## Citation
If you use this code in your research, please cite:

```bibtex
@software{nlp_ohca_classifier_v3,
    title={NLP OHCA Classifier v3.0: BERT-based Detection of Out-of-Hospital Cardiac Arrest with Enhanced Methodology},
    author={Mona Moukaddem},
    year={2025},
    url={https://github.com/monajm36/ohca-classifier-3.0},
    note={Enhanced methodology addressing data leakage, threshold optimization, and evaluation bias}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support
For questions or issues:
- Check the [Issues](https://github.com/monajm36/ohca-classifier-3.0/issues) page
- Create a new issue if needed
- Review examples in the `examples/` folder

## Methodology References
The v3.0 improvements are based on established machine learning best practices:
- Patient-level data splits prevent data leakage in healthcare AI
- Proper train/validation/test methodology ensures unbiased evaluation
- Optimal threshold finding improves clinical performance
- Larger sample sizes enhance model generalization

## Acknowledgments
- PubMedBERT model from Microsoft Research
- MIMIC-III dataset for model development
- Transformers library by Hugging Face
- PyTorch for deep learning framework
- Data science community for methodological guidance
