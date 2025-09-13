#!/usr/bin/env python3
"""
Data Preparation Helper for OHCA Classifier

This script helps prepare your data in the correct format for training or inference.
Supports both labeled training data and discharge notes for inference.
"""

import pandas as pd
import sys
import os
import argparse
from pathlib import Path

def validate_file_path(file_path):
    """Validate that file exists and is readable"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        raise ValueError(f"File must be a CSV file: {file_path}")

def suggest_column_mapping(df_columns, target_column):
    """Suggest best column mapping based on column names"""
    df_columns_lower = [col.lower() for col in df_columns]
    
    mapping_hints = {
        'hadm_id': ['hadm_id', 'admission_id', 'adm_id', 'id', 'admission', 'hadm'],
        'clean_text': ['clean_text', 'text', 'note_text', 'discharge_text', 'discharge_note', 
                      'note', 'discharge_summary', 'summary', 'notes', 'discharge'],
        'ohca_label': ['ohca_label', 'label', 'ohca', 'target', 'outcome', 'class', 
                      'classification', 'cardiac_arrest', 'arrest']
    }
    
    if target_column in mapping_hints:
        for hint in mapping_hints[target_column]:
            for i, col_lower in enumerate(df_columns_lower):
                if hint in col_lower:
                    return df_columns[i]
    
    return None

def prepare_labeled_data(input_path, output_path=None, interactive=True, column_mapping=None):
    """
    Prepare manually labeled data for training
    
    Args:
        input_path: Path to input CSV
        output_path: Path for output CSV (optional)
        interactive: Whether to prompt user for missing columns
        column_mapping: Dict mapping existing columns to required columns
    """
    print("Preparing labeled data for training...")
    
    # Validate input
    validate_file_path(input_path)
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")
    
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {list(df.columns)}")
    
    # Required columns for training
    required_cols = ['hadm_id', 'clean_text', 'ohca_label']
    final_mapping = {}
    
    # Use provided mapping if available
    if column_mapping:
        final_mapping.update(column_mapping)
    
    # Check for missing columns and suggest mappings
    for req_col in required_cols:
        if req_col not in df.columns and req_col not in final_mapping.values():
            # Try to find automatic mapping
            suggested = suggest_column_mapping(df.columns, req_col)
            
            if suggested and suggested not in final_mapping:
                if interactive:
                    print(f"\nColumn '{req_col}' not found.")
                    print(f"Suggested mapping: '{suggested}' -> '{req_col}'")
                    response = input("Accept this mapping? (y/n/custom): ").lower().strip()
                    
                    if response in ['y', 'yes']:
                        final_mapping[suggested] = req_col
                    elif response in ['custom', 'c']:
                        print(f"Available columns: {list(df.columns)}")
                        custom_col = input(f"Which column should be used for '{req_col}'? ")
                        if custom_col in df.columns:
                            final_mapping[custom_col] = req_col
                        else:
                            print(f"Warning: Column '{custom_col}' not found")
                    # else: skip this column
                else:
                    # Non-interactive mode: use suggestion
                    final_mapping[suggested] = req_col
                    print(f"Auto-mapped: '{suggested}' -> '{req_col}'")
            elif interactive:
                print(f"\nColumn '{req_col}' not found and no good suggestion available.")
                print(f"Available columns: {list(df.columns)}")
                mapped_col = input(f"Which column should be used for '{req_col}' (or 'skip')? ")
                if mapped_col in df.columns:
                    final_mapping[mapped_col] = req_col
                elif mapped_col.lower() != 'skip':
                    print(f"Warning: Column '{mapped_col}' not found")
    
    # Apply column mapping
    if final_mapping:
        df = df.rename(columns=final_mapping)
        print(f"Applied column mapping: {final_mapping}")
    
    # Validate required columns are now present
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after mapping: {missing_required}")
    
    # Add missing optional columns
    if 'subject_id' not in df.columns:
        df['subject_id'] = df['hadm_id']
        print("Added subject_id column (copied from hadm_id)")
    
    if 'confidence' not in df.columns:
        df['confidence'] = 4
        print("Added default confidence scores")
    
    # Validate and clean data
    initial_len = len(df)
    df = df.dropna(subset=['hadm_id', 'clean_text', 'ohca_label'])
    
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing required data")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # Validate ohca_label values
    try:
        df['ohca_label'] = pd.to_numeric(df['ohca_label'], errors='coerce')
        invalid_labels = df['ohca_label'].isna().sum()
        if invalid_labels > 0:
            print(f"Warning: {invalid_labels} invalid ohca_label values found, removing these rows")
            df = df.dropna(subset=['ohca_label'])
        
        df['ohca_label'] = df['ohca_label'].astype(int)
        unique_labels = sorted(df['ohca_label'].unique())
        
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"ohca_label must contain only 0 and 1, found: {unique_labels}")
            
    except Exception as e:
        raise ValueError(f"Error processing ohca_label column: {e}")
    
    # Set output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_prepared.csv")
    
    # Save prepared data
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving prepared data: {e}")
    
    # Summary
    ohca_count = (df['ohca_label'] == 1).sum()
    non_ohca_count = (df['ohca_label'] == 0).sum()
    
    print(f"\nLabeled data prepared successfully:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Total records: {len(df):,}")
    print(f"  OHCA cases (label=1): {ohca_count:,}")
    print(f"  Non-OHCA cases (label=0): {non_ohca_count:,}")
    print(f"  OHCA prevalence: {ohca_count/len(df)*100:.1f}%")
    print(f"  Final columns: {list(df.columns)}")
    
    return output_path

def prepare_discharge_notes(input_path, output_path=None, interactive=True, column_mapping=None):
    """
    Prepare discharge notes for inference
    
    Args:
        input_path: Path to input CSV
        output_path: Path for output CSV (optional)
        interactive: Whether to prompt user for missing columns
        column_mapping: Dict mapping existing columns to required columns
    """
    print("Preparing discharge notes for inference...")
    
    # Validate input
    validate_file_path(input_path)
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")
    
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {list(df.columns)}")
    
    # Required columns for inference
    required_cols = ['hadm_id', 'clean_text']
    final_mapping = {}
    
    # Use provided mapping if available
    if column_mapping:
        final_mapping.update(column_mapping)
    
    # Check for missing columns
    for req_col in required_cols:
        if req_col not in df.columns and req_col not in final_mapping.values():
            # Try to find automatic mapping
            suggested = suggest_column_mapping(df.columns, req_col)
            
            if suggested and suggested not in final_mapping:
                if interactive:
                    print(f"\nColumn '{req_col}' not found.")
                    print(f"Suggested mapping: '{suggested}' -> '{req_col}'")
                    response = input("Accept this mapping? (y/n/custom): ").lower().strip()
                    
                    if response in ['y', 'yes']:
                        final_mapping[suggested] = req_col
                    elif response in ['custom', 'c']:
                        print(f"Available columns: {list(df.columns)}")
                        custom_col = input(f"Which column should be used for '{req_col}'? ")
                        if custom_col in df.columns:
                            final_mapping[custom_col] = req_col
                        else:
                            print(f"Warning: Column '{custom_col}' not found")
                else:
                    # Non-interactive mode: use suggestion
                    final_mapping[suggested] = req_col
                    print(f"Auto-mapped: '{suggested}' -> '{req_col}'")
            elif interactive:
                print(f"\nColumn '{req_col}' not found and no good suggestion available.")
                print(f"Available columns: {list(df.columns)}")
                mapped_col = input(f"Which column should be used for '{req_col}' (or 'skip')? ")
                if mapped_col in df.columns:
                    final_mapping[mapped_col] = req_col
                elif mapped_col.lower() != 'skip':
                    print(f"Warning: Column '{mapped_col}' not found")
    
    # Apply column mapping
    if final_mapping:
        df = df.rename(columns=final_mapping)
        print(f"Applied column mapping: {final_mapping}")
    
    # Validate required columns are now present
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after mapping: {missing_required}")
    
    # Clean data
    initial_len = len(df)
    df = df.dropna(subset=['hadm_id', 'clean_text'])
    
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing required data")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # Ensure proper data types
    df['hadm_id'] = df['hadm_id'].astype(str)
    df['clean_text'] = df['clean_text'].astype(str)
    
    # Check for empty text
    empty_text = df['clean_text'].isin(['', 'nan', 'None', 'null']).sum()
    if empty_text > 0:
        print(f"Warning: {empty_text} cases have empty/null text")
    
    # Set output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_prepared.csv")
    
    # Save prepared data
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving prepared data: {e}")
    
    print(f"\nDischarge notes prepared successfully:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Records: {len(df):,}")
    print(f"  Final columns: {list(df.columns)}")
    
    return output_path

def main():
    """Main function for console script entry point"""
    parser = argparse.ArgumentParser(
        description='Prepare data for OHCA classifier training or inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare labeled training data
  ohca-prepare labeled training_data.csv
  
  # Prepare discharge notes for inference
  ohca-prepare discharge notes.csv
  
  # Non-interactive mode with custom output
  ohca-prepare labeled data.csv --output prepared_data.csv --no-interactive
  
  # With custom column mapping
  ohca-prepare labeled data.csv --map admission_id:hadm_id --map note:clean_text
        """
    )
    
    parser.add_argument('data_type', choices=['labeled', 'discharge'],
                       help='Type of data to prepare')
    parser.add_argument('input_path', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Run in non-interactive mode (use auto-mapping)')
    parser.add_argument('--map', action='append', dest='mappings',
                       help='Column mapping in format "source:target" (can be used multiple times)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return 1
    
    # Parse column mappings
    column_mapping = {}
    if args.mappings:
        for mapping in args.mappings:
            if ':' not in mapping:
                print(f"Error: Invalid mapping format '{mapping}'. Use 'source:target'")
                return 1
            source, target = mapping.split(':', 1)
            column_mapping[source.strip()] = target.strip()
    
    try:
        if args.data_type == "labeled":
            output_path = prepare_labeled_data(
                args.input_path, 
                args.output, 
                interactive=not args.no_interactive,
                column_mapping=column_mapping
            )
        elif args.data_type == "discharge":
            output_path = prepare_discharge_notes(
                args.input_path, 
                args.output, 
                interactive=not args.no_interactive,
                column_mapping=column_mapping
            )
        
        print(f"\nSuccess! Prepared data saved to: {output_path}")
        
        if args.verbose:
            # Show file info
            prepared_df = pd.read_csv(output_path)
            print(f"\nPrepared file details:")
            print(f"  Rows: {len(prepared_df):,}")
            print(f"  Columns: {list(prepared_df.columns)}")
            print(f"  File size: {os.path.getsize(output_path)} bytes")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
