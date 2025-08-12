from evaluationContainer import *

_path_to_data = r"valid_output.xlsx"
_export = "results_tolerance_0.0.xlsx"

_my_evaluation = evaluation_container(predictions_xlsx = _path_to_data)

_my_evaluation.obtain_thresholds_with_tolerance(tolerance=0.0)
_scores_dict = _my_evaluation.obtain_scores(threshold = None, output_xlsx = _export)
print(_scores_dict)

