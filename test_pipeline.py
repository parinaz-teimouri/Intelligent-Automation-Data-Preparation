from data_preparation import AutoDataPreparer
import pandas as pd
import sys
import os

def test_pipeline(file_path):
    print(f"Starting pipeline for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Create preparer instance with optional target column for evaluation
    target_col = None
    if "complex_data.csv" in file_path:
        target_col = 'membership_status' 
    elif "regression_data.csv" in file_path:
        target_col = 'house_price'
    elif "global_freelancers_raw.csv" in file_path:
        target_col = 'rating'
    
    preparer = AutoDataPreparer(file_path, target_col=target_col)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"processed_{base_name}.csv"
    
    # Run the full pipeline
    try:
        preparer.run_pipeline(output_path)
        
        # Load and check the processed data
        if os.path.exists(output_path):
            processed_data = pd.read_csv(output_path)
            print("\nProcessed Data Preview (ML Ready):")
            print(processed_data.head())
        print(f"\nPipeline completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if a file path is provided as a command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default to dummy_data.csv if it exists, otherwise ask user for input
        if os.path.exists('dummy_data.csv'):
            input_file = 'dummy_data.csv'
        else:
            print("Usage: python test_pipeline.py <path_to_your_data_file>")
            sys.exit(1)
            
    test_pipeline(input_file)
