from evaluationContainer import *

# Data
_xlsx_fils_dir = r"Labeled"
_threhsold_xlsx = "results_tolerance_0.0.xlsx"

# Create xlsx with predictions
generate_combined_xlsx(input_dir = _xlsx_fils_dir,
              output_xlsx = "HIP.xlsx")

# Create conatiner
_my_evaluation = evaluation_container(predictions_xlsx = "HIP.xlsx")#"valid_output.xlsx")#"HIP.xlsx")

# Obtain threhsolds
#_my_evaluation.obtain_thresholds_with_tolerance(tolerance=0.0)
_my_evaluation.obtain_threshold_from_xlsx(_threhsold_xlsx)

# Predict scores
#_scores_dict = _my_evaluation.obtain_scores(threshold = None, output_xlsx = _export)
_my_evaluation.obtain_in_depth_scores(threshold = None, output_dir = "HIP_results_0.0")
#print(_scores_dict)


