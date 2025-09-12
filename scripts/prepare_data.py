#!/usr/bin/env python3
"""
Data Preparation Helper for OHCA Classifier

This script helps prepare your data in the correct format for training or inference.
"""

import pandas as pd
import sys

def prepare_labeled_data(input_path, output_path=None):
    """Prepare manually labeled data for training"""
    print("Preparing labeled data for training...")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Interactive column mapping
    required_cols = ['hadm_id', 'clean_text', 'ohca_label']
    column_mapping = {}
    
    for req_col in required_cols:
        if req_col not in df.columns:
            print(f"\nColumn '{req_col}' not found.")
            print(f"Available columns: {list(df.columns)}")
            mapped_col = input(f"Which column should be used for '{req_col}'? ")
            if mapped_col in df.columns:
                column_mapping[mapped_col] = req_col
            else:
                print(f"Column '{mapped_col}' not found. Skipping...")
    
    # Apply mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Applied column mapping: {column_mapping}")
    
    # Add missing optional columns
    if 'subject_id' not in df.columns:
        df['subject_id'] = df['hadm_id']
        print("Added subject_id column (copied from hadm_id)")
    
    if 'confidence' not in df.columns:
        df['confidence'] = 4
        print("Added default confidence scores")
    
    # Validate and clean
    df = df.dropna(subset=['hadm_id', 'clean_text', 'ohca_label'])
    
    # Set output path
    if output_path is None:
        base_name = input_path.replace('.csv', '')
        output_path = f"{base_name}_prepared.csv"
    
    df.to_csv(output_path, index=False)
    
    print(f"\nData prepared successfully:")
    print(f"  Output: {output_path}")
    print(f"  Records: {len(df)}")
    print(f"  OHCA cases: {(df['ohca_label']==1).sum()}")
    print(f"  Columns: {list(df.columns)}")

def prepare_discharge_notes(input_path, output_path=None):
    """Prepare discharge notes for inference"""
    print("Preparing discharge notes for inference...")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Interactive column mapping
    required_cols = ['hadm_id', 'clean_text']
    column_mapping = {}
    
    for req_col in required_cols:
        if req_col not in df.columns:
            print(f"\nColumn '{req_col}' not found.")
            print(f"Available columns: {list(df.columns)}")
            mapped_col = input(f"Which column should be used for '{req_col}'? ")
            if mapped_col in df.columns:
                column_mapping[mapped_col] = req_col
    
    # Apply mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Applied column mapping: {column_mapping}")
    
    # Clean data
    df = df.dropna(subset=['hadm_id', 'clean_text'])
    
    # Set output path
    if output_path is None:
        base_name = input_path.replace('.csv', '')
        output_path = f"{base_name}_prepared.csv"
    
    df.to_csv(output_path, index=False)
    
    print(f"\nDischarge notes prepared:")
    print(f"  Output: {output_path}")
    print(f"  Records: {len(df)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/prepare_data.py labeled <input.csv>     # For training data")
        print("  python scripts/prepare_data.py discharge <input.csv>   # For inference data")
        sys.exit(1)
    
    data_type = sys.argv[1]
    input_path = sys.argv[2]
    
    if data_type == "labeled":
        prepare_labeled_data(input_path)
    elif data_type == "discharge":
        prepare_discharge_notes(input_path)
    else:
        print("Data type must be 'labeled' or 'discharge'")
        sys.exit(1)
