import os
import torch
from .models_NMDD_BCH_SAROS import *
import cv2
import json
from .Containers import *
import time

class oracle():
    """
    Class for predictions of the models on the images and CTs
    """
    def __init__(self, 
                 oracle_model_type: str = None, 
                 oracle_model_path: str = None,
                 oracle_device: str = 'cpu',
                 oracle_scream: bool = False,
                 oracle_number_of_tools: int = None,
                 oracle_self_esteem: str = None,
                 ):
        """
        Init of the orcale. It requires paths to the model which is going to be loaded

        Args:
            * oracle_model_type, str, string which describes model. There are several models currently implemented
            * oracle_model_path, str, path to the model weights (where model has been saved)
            * oracle_device, str, cpu default or cuda:n if model and data should be predicted on gpu
            * oracle_screm, bool, if true the verbose option is on.
            * oracle_number_of_tools, int, number of processes used when exporting series from CT to 2D
            * oracle_self_esteem, str, path to the json file containing thresholds for model prediction. If None is provided
                                 then default of 0.5 for each body part will be used.
        """
        # Set verbose
        self.verbose = oracle_scream

        # Set processes
        self.multi_process = oracle_number_of_tools

        # Set device
        self.oracle_device = oracle_device

        # Load model
        self.oracle_model = self.__load_model(type = oracle_model_type, 
                                              path = oracle_model_path)
        
        # Load aug model 
        self.aug_model = self.__load_aug_model(type = 'GRAY_Simple')

        # Define names
        self.names = ["skull", "shoulder", "humerus", "vertebrae_C", 
                      "thorax", "vertebrae_L", "forearm", "pelvis", 
                      "femur", "hand", "patella", "shin", "tarsal", 
                      "foot"]
        
        # Load thresholds
        self.thresholds_dict = self.__load_thresholds(path = oracle_self_esteem)
        
    
    def __load_thresholds(self, path:str = None)->dict:
        """
        Method which returns thresholds obtained by the provided file path. If the path is
        not provided, then the "defualt" threshold (0.5 for each body part) is returned

        Args:
            * path, str, path to json file conatiainging the thresholds for each body part

        Returns:
            * dict, dictionary containing all body parts and their thresolds
        """

        if path != None:
            # Case where dict path is provided
            assert os.path.exists(path), f"Json path is invalid. Could not find {path}! Please check it out"
            # Load dict
            with open(path, 'r') as _json_file:
                _threshold_dict = json.load(_json_file)
        else:
            _threshold_dict = {}
            # Case where the default dict must be generated
            for _body_part in self.names:
                _threshold_dict[_body_part] = 0.5

        # Return
        return _threshold_dict

    def __preprocess_data(self, 
                          image_path: str = None,
                          image_size: int = 224)->torch.Tensor:
        """
        Method which preprocess data. It loads image from the given path and applies all 
        necessary augmentations.

        Args:
            * case_path, str, path to the input image on which the prediction will be done.
            * image_size, 
        Return:
            * returns preprocessed tensor ready to be used as an input to the model
        """
        # Image
        # Safty
        while(1):
            _image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if _image is not None:
                break
    
        # Generate image dimension
        _old_size = _image.shape[:2] # old_size is in (height, width) format
        
        # Calculate new size
        _ratio = float(image_size)/max(_old_size)
        _new_size = tuple([int(_x*_ratio) for _x in _old_size])

        # new_size should be in (width, height) format
        _image = cv2.resize(_image, (_new_size[1], _new_size[0]))
        
        # Calculate padding
        _delta_w = image_size - _new_size[1]
        _delta_h = image_size - _new_size[0]
        _top, _bottom = _delta_h//2, _delta_h-(_delta_h//2)
        _left, _right = _delta_w//2, _delta_w-(_delta_w//2)
        
        # Pad
        color = [0]
        _image = cv2.copyMakeBorder(_image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=color)
        
        # Export to tensor
        _image = torch.from_numpy(_image)
        _image = _image.to(torch.float32)
        _image /= 255.0
        
        # Add batch
        _image = _image.unsqueeze(0)

        # Send it to device
        _image = _image.to(self.oracle_device)

        # Applay augumentation model  
        _image = self.aug_model(_image)

        # Return
        return _image
       
    def __load_model(self,
                     type: str = None, 
                     path: str = None)->torch.nn.Module:
        """
        Method which loads the model and returns it with its weights

        Args:
            * type, str, type of the model. Currently implemented: dense161
            * path, str, path to the model's weights

        Returns:
            * model, torch.nn.Module which is cappable to predict data.
        """
        # Check if path to the mmodel exist and if type is correct
        assert os.path.exists(path), f"Could not find path to the model's weights: {path}! Please check the path"
        assert type in ['dense161'], f"Given model has not been implemented: {type}! Check docstring to find list of implemented models"

        # Init model
        if type == 'dense161':
            _model = DenseNet161(pretrained = False, number_of_classes = 14)

        # Load weights
        _state_dict = torch.load(path, map_location = self.oracle_device)
        _model.load_state_dict(_state_dict['model_state'])
        _model.name = _state_dict['optimizer_name']
        
        # Transfer model to device
        _model = _model.to(self.oracle_device)

        # Set it to eval mode
        _model.eval()

        # Info
        if self.verbose:
            print(f"Loaded model of type: {type} and loaded its weights. Model is set on device: {self.oracle_device}!")
                
        # Return
        return _model
    
    def __load_aug_model(self,
                         type: str = None)->torch.nn.Module:
        """
        Method to load augumentation model. Based on the type different models are available

        Args:
            * type, str, type of the augumentation method
        
        Returns:
            * Augumentation model of choice set on the device given by self.oracle_device
        """
        if type == 'GRAY_Simple':
            _aug_model = TransformToGray_Simple()
            return _aug_model
    
    def __obtain__prediction(self,  
                             image_path: str = None,
                             image_size: int = 224):
        """
        Method which obtains model prediction for given image.

        Args:
            * image_path, str, path to the image whic is going to be predicted
            * image_size, str, size of the image which will be feed to the model

        Returns: 
            * dict contataining exported predictions

        Export:
            * Saves prediction in "prediction.json" in the same dir where the image is
            located

        """
        if self.verbose:
            print("--------------------------------------------------------------------------------------")
            print(f"Preprocess data for model")
        _preprocessed_image = self.__preprocess_data(image_path = image_path,
                                                     image_size = image_size)
        # Obtain prediction
        # Info
        if self.verbose:
            print(f"Obtaining predictions!")
        _prediction = self.oracle_model(_preprocessed_image)        
        _prediction_numpy = _prediction.to("cpu").detach().numpy().flatten()
        _output_dict = {self.names[_i]: int(_prediction_numpy[_i] >= self.thresholds_dict[self.names[_i]]) for _i in range(len(self.names))}
        _probability_dict = {self.names[_i]: float(_prediction_numpy[_i]) for _i in range(len(self.names))}

         # Export
        if self.verbose:
            print(f"Exporting")
            print("--------------------------------------------------------------------------------------")

        # Convert the dictionary to a JSON string
        _json_output_string = json.dumps([_output_dict, _probability_dict], indent=4)

        # Export json
        _dir = os.path.dirname(image_path)
        _image_name = os.path.basename(image_path).split(".png")[0]
        _name = os.path.join(_dir, f"{_image_name}_prediction.json")
        with open(_name, "w") as _json_file:
            _json_file.write(_json_output_string)

        # Return
        return _output_dict

    def __export_CT(self,
                    input_path: str = None,
                    id: str = None,
                    export_path: str = None)->list:
        """
        Method which exports dcms to images (CT2D). This script is strongly related to script exportOneFolder.py

        Args:
            * input_path, str, path to the image whic is going to be predicted
            * id, str, id of the case, if the id is left to None, then the basedir of the image_path will be 
              set as the image's id
            * export_path, str, if left none, then in image_path directory will be created nifti dir with all necessary subdirs.
              In case that given image_path points to the image, then the new json will be created solly based on the image name with
              extension ".json".

        Retruns:
            * list of image paths

        """
        # Obtain ID
        if id == None:
            # Scenario 1
            _id = os.path.basename(input_path)
        else:
            # Scenario 2 
            _id = id
        
        # Obtain Export path
        if export_path == None:
            # Scenario 1
            _export_path = os.path.join(input_path, "nifti")
        else:
            # Scenario 2
            _export_path = export_path
           
        # Remove the directory and its contents
        if os.path.exists(_export_path):
            shutil.rmtree(_export_path)
        os.makedirs(_export_path)

        # Info
        if self.verbose:
            print(f"Created export dir: {_export_path}!")
        
        # Create container
        # Info
        if self.verbose:
            print(f"Extracting DCM series!")
        _container = SeriesContainer(verbose = self.verbose, 
                                     number_of_processes = self.multi_process)
        _container.obtain_data_from_series(path_dir = input_path, mode='o')

        # Extract data
        if self.verbose:
            print(f"Process started exporting data!")
        _container.export_data_to_directory(export_dir_path = _export_path,
                                            export_id = _id,
                                            export_nifti = True, 
                                            export_npy = False )      
        if self.verbose:
            print(f"Process finished exporting data!")
        
        if self.verbose:
            print(f"Process started exporting images!")
        _container.export_images_to_directory(export_dir_path = _export_path,
                                              export_id = _id,
                                              resample_spacing = [0.5, 0.5, 0.5])
        if self.verbose:
            print(f"Process finished exporting images !")
            print(f"Process finished!")
        
        # List
        _number_of_cases = len(_container)
        _number_of_files = 4 + int(1) + int(0) #Nifti and npy
        # Wait until everything is saved
        while(1):
            _export_list = os.listdir(_export_path)
            if len(_export_list) == _number_of_cases * _number_of_files:
                _export_list = [os.path.join(_export_path, _f) for _f in os.listdir(_export_path) if _f.endswith('_reducted_image.png')]
                break
        return _export_list

    def reorganize_cases(self,
                         export_path: str = None):
        """
        Method which reorganize export path files into "id_case" subfolders and where every sub folder
        contains all files connected to that id-case

        Args:
            *  export_path, str, must not be left none. It must be the folder which has all id_cases in that dir a folder with subfolder id_case
            will be created.
        """
        # Check if path exists
        assert os.path.exists(export_path), f"Path does not exists {export_path}, check it and try again."
        _root_dir = export_path

        # Scan for file names and obtain all id_case pairs
        _id_case_list = []
        for _dirpath, _, _filenames in os.walk(export_path):
            # Filter filenames that end with .nii.gz
            _id_case_list.extend(_file.split(".nii.gz")[0] for _file in _filenames if _file.endswith('.nii.gz'))

        # Info
        if self.verbose:
            print(f"Found {len(_id_case_list)} cases!")
        
        # Create dirs and copy files
        for _case in _id_case_list:
            # Create dir
            _final_dir_path = os.path.join(export_path, _case)
            os.makedirs(_final_dir_path)
            
            # Check if file exists
            while os.path.exists(_final_dir_path) == False:
                continue
            
            # Cut all files to related to _case in newly created dir
            ## Get file paths
            _files_assigned_to_case = []
            for _dirpath, _, _filenames in os.walk(export_path):
                # Filter filenames that end with .nii.gz
                _files_assigned_to_case.extend(os.path.join(_dirpath, _file) for _file in _filenames if _case in _file)

            ## Move them
            for _file in _files_assigned_to_case:
                # Extract the file name from the file path
                _file_name = os.path.basename(_file)
                # Check if file exists
                while os.path.exists(_file) == False:
                    continue
                # Move
                shutil.move(_file, os.path.join(_final_dir_path, _file_name))
            

    def predict(self,
                input_path: str = None,
                image_size: int = 224,
                id: str = None,
                export_path: str = None)->dict:
        """
        Predict image with loaded model for a given image path.
        
        Args:
            * input_path, str, path to the image or folder which is going to be predicted
            * image_size, str, size of the image which will be feed to the model
            * id, str, id of the case, if the id is left to None, then the basedir of the image_path will be 
              set as the image's id
            * export_path, str, if left none, then in image_path directory will be created nifti dir with all necessary subdirs.
              In case that given image_path points to the image, then the new json will be created solly based on the image name with
              extension ".json".

        Retruns:
            * dict of dicts, dictionary conatining dictionaries where key is image, and value is its predictions.

        Export:
            * export predictions to json in export_path (if provided) 
        """
        # Create storage
        _output_dict = {}

        # Check if given path is dir, if it is - than the images are going to be extracted in that dir
        if os.path.isdir(input_path):
            # Info
            if self.verbose:
                print(f"Extracting and exporting data to dirs!")

            # Export dir to files
            _images_paths = self.__export_CT(input_path = input_path,
                             id = id,
                             export_path = export_path)
            

            for _path in _images_paths:
                _output_dict[_path] = self.__obtain__prediction(image_path = _path,
                                                                image_size = image_size)

        # Chech if it is image file
        if os.path.isfile(input_path) and input_path.endswith('.png'):
            _output_dict[input_path] = self.__obtain__prediction(image_path = input_path,
                                                     image_size = image_size)
        
        # Obtain Export path
        if export_path == None:
            # Scenario 1
            _export_path = os.path.join(input_path, "nifti", "predictions.json")
        else:
            # Scenario 2
            _export_path = os.path.join(export_path, "predictions.json")
        
        # Export json to file.
        with open(_export_path, 'w') as _json_file:
            json.dump(_output_dict, _json_file, indent=4)
        
        # Return
        return _output_dict

   