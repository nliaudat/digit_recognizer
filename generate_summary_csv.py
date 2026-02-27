import pandas as pd
import os
import glob
from datetime import datetime

EXPORTS_DIR = "exported_models"
current_date_str = datetime.now().strftime("%d.%m.%Y")
SUMMARY_CSV = os.path.join(EXPORTS_DIR, f"summary_model_comparison_{current_date_str}.csv")

def get_scenario(directory):
    directory = directory.upper()
    cls_part = "100cls" if "100CLS" in directory else "10cls"
    color_part = "GRAY" if "GRAY" in directory else "RGB"
    return f"{cls_part}_{color_part}"

def create_summary():
    # Find all model_comparison.csv files
    search_pattern = os.path.join(EXPORTS_DIR, "*", "test_results", "model_comparison.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"Error: No model_comparison.csv files found in {EXPORTS_DIR} subdirectories.")
        return

    all_dfs = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            # Add file modification date just in case we need it for sorting latest
            mod_time = os.path.getmtime(f)
            temp_df['Date'] = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            all_dfs.append(temp_df)
        except Exception as e:
            print(f"Could not read {f}: {e}")
            
    if not all_dfs:
        print("No valid data could be aggregated.")
        return
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up model names (remove .tflite)
    df['Model Name'] = df['Model'].apply(lambda x: str(x).replace('.tflite', ''))
    
    # Determine scenario
    df['Scenario'] = df['Directory'].apply(get_scenario)
    
    # Sort by Date so we can keep the latest entry for each model-scenario pair
    df = df.sort_values('Date')
    df = df.drop_duplicates(subset=['Model Name', 'Scenario'], keep='last')
    
    metrics = ['Parameters', 'Size_KB', 'Accuracy', 'Inference_Time_ms', 'Inferences_per_second', 'Tested_Images']
    
    # Pivot the dataframe
    pivot_df = df.pivot(index='Model Name', columns='Scenario', values=metrics)
    
    # Flatten the multi-level columns
    # We want columns like: 10cls_GRAY Parameters, 10cls_GRAY Size_KB
    pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
    
    # Let's specify the order of scenarios and metrics
    scenarios = ['10cls_GRAY', '10cls_RGB', '100cls_GRAY', '100cls_RGB']
    ordered_cols = []
    for scenario in scenarios:
        for metric in metrics:
            col_name = f"{scenario} {metric}"
            if col_name in pivot_df.columns:
                ordered_cols.append(col_name)
                
    pivot_df = pivot_df[ordered_cols].reset_index()
    
    # Fill NaN with empty string or 'N/A'
    pivot_df = pivot_df.fillna('N/A')
    
    # Sort by 10cls_GRAY Accuracy if available, else by Model Name
    if '10cls_GRAY Accuracy' in pivot_df.columns:
        # Convert to numeric for sorting, putting N/A at the bottom
        pivot_df['_sort'] = pd.to_numeric(pivot_df['10cls_GRAY Accuracy'], errors='coerce')
        pivot_df = pivot_df.sort_values(['_sort', 'Model Name'], ascending=[False, True])
        pivot_df = pivot_df.drop(columns=['_sort'])
    else:
        pivot_df = pivot_df.sort_values('Model Name')
    
    # Save to CSV
    pivot_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary CSV created at: {SUMMARY_CSV}")

if __name__ == "__main__":
    create_summary()
