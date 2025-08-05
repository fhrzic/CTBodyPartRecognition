import pandas as pd
import json

# Input to the xlsx which has Thresholds tab in it
_input_xlsx = r"C:\Users\CH258598\Desktop\FiteredDatasetNNMD256Subset\EvaluatePipeline\results_tolerance_0.0.xlsx"

# Output json file
_output_json = "best_threshold.json"

# Read data
_df = pd.read_excel(_input_xlsx, sheet_name = "Thresholds", index_col = 0)

# Obtain dict
_data_dict = _df[0].to_dict()

# Export it to json
with open(_output_json, 'w') as json_file:
    json.dump(_data_dict, json_file, indent=4)



