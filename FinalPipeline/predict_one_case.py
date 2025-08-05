from Utils.oracle import *
import time

# Define model
_model_type = "dense161"
_model_weights = "/mnt/SSD/Franko/Desktop/FinalPipeline/BestModel.pth"
_input_size = 224

# Define input 
#_image_path = "/home/franko/Desktop/FilteredDataNNMD256Subset/Predict_NMDD_BCH_SAROS/case-default__2.16.840.114379.3000.409.20181231.1055618.0.2_reducted_image.png"
#_folder_path = "/home/franko/Desktop/20002289"
_path = "/mnt/HDD/CT"
_paths =[os.path.join(_path, _f) for _f in os.listdir(_path)]

_folder_path = _paths[0]


for _folder_path in _paths:
    print(f"Working on {_folder_path}")
    # Record the start time
    _start_time = time.time()
    
    # Create container and obtain prediction
    _oracle = oracle(oracle_model_type = _model_type, 
                    oracle_model_path = _model_weights, 
                    oracle_device = "cpu",
                    oracle_scream = False,
                    oracle_number_of_tools = 24,
                    oracle_self_esteem = "best_threshold.json")

    _prediction = _oracle.predict(input_path = _folder_path,
                                image_size = _input_size,
                                id = None,
                                export_path = os.path.join(_folder_path, "nifti"))

    _oracle.reorganize_cases(export_path = os.path.join(_folder_path, "nifti"))

    # Record the end time
    _end_time = time.time()

    # Calculate the elapsed time
    _elapsed_time = _end_time - _start_time

    print(f"DONE!")
    print(f"Elapsed time: {_elapsed_time:.4f} seconds")


