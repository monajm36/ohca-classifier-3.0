# OHCA Classifier Examples

## Quick Start for Real Data

### 1. Train on Your Labeled Data
```bash
# If your data needs preparation:
python ../scripts/prepare_data.py labeled your_labeled_data.csv

# Train the model:
python ../scripts/train_from_labeled_data.py your_prepared_data.csv
```

### 2. Apply to New Discharge Notes
```bash
# If your data needs preparation:
python ../scripts/prepare_data.py discharge your_discharge_notes.csv

# Apply the model:
python ../scripts/predict_ohca.py ./trained_ohca_model your_prepared_discharge_notes.csv
```

## Command Line Usage

### Training
```bash
python scripts/train_from_labeled_data.py data.csv --model_path ./my_model --epochs 5
```

### Prediction
```bash
python scripts/predict_ohca.py ./my_model discharge_notes.csv --output predictions.csv
```

## Data Format Requirements

### For training (labeled data):
- `hadm_id`: Admission ID
- `clean_text`: Discharge note text
- `ohca_label`: 1 for OHCA, 0 for non-OHCA
- `subject_id`: Patient ID (optional)

### For prediction (discharge notes):
- `hadm_id`: Admission ID
- `clean_text`: Discharge note text

## Demo Examples

Run the included demo examples:
```bash
python training_example.py    # Shows v3.0 methodology
python inference_example.py  # Shows inference workflow
```
