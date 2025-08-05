import os
import SimpleITK as sitk
from .Conversion import *
from .Containers import *
from einops import rearrange
from scipy.ndimage import label
import json
import numpy as np
import time
import pandas as pd
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import math


class patch_container():
    """
    Class for extracting patches from the niftii file
    """
        
    def __init__(self, input_dir:str = None):
        """
        Init function. 

        Args:
            * input_dir, str, path to the input dir
        """
        # Check if dir exists
        assert os.path.exists(input_dir), f"Missing directory {input_dir}"
        
        # Assign dir
        self.input_dir = input_dir

        # Define remap dictionary
        self.remap_dict =  {
            91: [1, 1, 1],
            73: [2.1, 2, 2],
            74: [2.2, 3, 2],
            71: [3.1, 4, 2],
            72: [3.2, 5, 2],
            69: [4.1, 6, 3],
            70: [4.2, 7, 3],
            50: [5.1, 8, 4],
            49: [5.2, 9, 4],
            48: [5.3, 10, 4],
            47: [5.4, 11, 4],
            46: [5.5, 12, 4],
            45: [5.6, 13, 4],
            44: [5.7, 14, 4],
            116:[6, 15, 5],
            92: [7.1, 16, 5],
            104: [7.2, 17, 5],
            93: [8.1, 18, 5],
            105: [8.2, 19, 5],
            94: [9.1, 20, 5],
            106: [9.2, 21, 5],
            95: [10.1, 22, 5],
            107: [10.2, 23, 5],
            96: [11.1, 24, 5],
            108: [11.2, 25, 5],
            97: [12.1, 26, 5],
            109: [12.2, 27, 5],
            98: [13.1, 28, 5],
            110: [13.2, 29, 5],
            99: [14.1, 30, 5],
            111: [14.2, 31, 5],
            100: [15.1, 32, 5],
            112: [15.2, 33, 5],
            101: [16.1, 34, 5],
            113: [16.2, 35, 5],
            102: [17.1, 36, 5],
            114: [17.2, 37, 5],
            103: [18.1, 38, 5],
            115: [18.2, 39, 5],
            43: [19.1, 40, 5],
            42: [19.2, 41, 5],
            41: [19.3, 42, 5],
            40: [19.4, 43, 5],
            39: [19.5, 44, 5],
            38: [19.6, 45, 5],
            37: [19.7, 46, 5],
            36: [19.8, 47, 5],
            35: [19.9, 48, 5],
            34: [19.11, 49, 5],
            33: [19.12, 50, 5],
            32: [19.13, 51, 5],
            31: [20.1, 52, 6],
            30: [20.2, 53, 6],
            29: [20.3, 54, 6],
            28: [20.4, 55, 6],
            27: [20.5, 56, 6],
            26: [21, 57, 8],
            124: [22, 58, 7],
            125: [23, 59, 7],
            77: [24.1, 60, 8],
            78: [24.2, 61, 8],
            25: [25, 62, 8],
            75: [26.1, 63, 9],
            76: [26.2, 64, 9],
            126: [27, 65, 10],
            127: [28, 66, 10],
            128: [29, 67, 10],
            118: [30, 68, 11],
            119: [31, 69, 12],
            120: [32, 70, 12],
            121: [33, 71, 13],
            122: [34, 72, 14],
            123: [35, 73, 14],
        }
        
        # Define name dictionary:
        self.cheatsheet_names = "cheatsheet_names.xlsx"

        # Define merging dictionary
        self.main_blobs_storage = None
        self.raw_blobs = None
        self.merged_blobs = None
        self.merging_critera = None
        self.total_segmentation_volume = None
        self.biggest_blobs = None
        self.filtered_blobs = None

        # Obtain important variables
        self.__obtain_data()

        # Read volumes and resample them
        self.__read_main_volumes()
        

    def refine_segmentation_volumes(self, export_path: str = None):
        """
        Method which refines segmentation volumes, merges them in one and creates merged labels.
        Support export of all found biggest blobs to json. 

        Args:
            * export_path, str, path to the export_json_storage. Only path to dir, name is default: 
            "raw_blobs.json"
        """
        # Obtain numpy arrays
        _total_segmentation_array = sitk.GetArrayFromImage(self.total_segmentation_volume)
        _appendicular_segmentation_array = sitk.GetArrayFromImage(self.appendicular_segmentation_volume)
        _appendicular_segmentation_array = rearrange(_appendicular_segmentation_array, 'z y x -> x y z')        
        _total_segmentation_array = rearrange(_total_segmentation_array, 'z y x -> x y z')        

        # Do a label shift for appendicular
        _appendicular_segmentation_array += 117
        # Find which labels are present
        _total_segmentation_labels = np.unique(_total_segmentation_array)
        _appendicular_segmentation_labels = np.unique(_appendicular_segmentation_array)

        # Remove overlaping label
        _appendicular_segmentation_labels = np.delete(_appendicular_segmentation_labels, 
                                                      np.where(_appendicular_segmentation_labels == 117))
       
        # Obtain blobs
        ## Structure element for seeking blobs
        _structure = [[[1 for _ in range(3)] for _ in range(3)] for _ in range(3)]
        _storage_dict = {}
        for _i, _key in enumerate(self.remap_dict):
            print(f"{_i+1}/{len(list(self.remap_dict.keys()))}")
            # Create binary array with 1 where the value is present
            if _key in _total_segmentation_labels:
                _binary_masked_array = (_total_segmentation_array == _key).astype(int)
            else:
                if _key in _appendicular_segmentation_labels:
                    _binary_masked_array = (_appendicular_segmentation_array == _key).astype(int)
                else:
                    continue

            # Obtain blobs
            _labeled_array, _num_features = label(input = _binary_masked_array, 
                                                  structure = _structure)
            
            # Calculate blobs
            _blobs = []
            for _feature_num in range(1, _num_features + 1):
                _blob = np.where(_labeled_array == _feature_num)
                _blobs.append(list(zip(_blob[0], _blob[1], _blob[2])))
        
            # Obtain the biggest blob
            
            if _blobs != []:
                _biggest_blob_index = [len(_blob) for _blob in _blobs]
                _biggest_blob_index = _biggest_blob_index.index(max(_biggest_blob_index))
                _biggest_blob = _blobs[_biggest_blob_index]

                # Convert everything to int (json req)
                _biggest_blob = [tuple(map(int, _tup)) for _tup in _biggest_blob]
                # Store it
                _storage_dict[_key] = list(_biggest_blob)
            
        # Export it to json
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        if export_path != None:
            with open(os.path.join(export_path, f"{_name}_biggest_blobs.json"), "w") as _outfile:
                json.dump(_storage_dict, _outfile)

        # Create variable
        self.raw_blobs = _storage_dict
    
    def obtain_raw_blobs(self, export_path: str = None):
        """
        Method which exports raw blobs to folder. It extracts for each label of interest 
        its indices in the volume and stores them in the export_path.

        Args:
            * export_path, str, if provided, path to where "merged_blobs.json" will be created
        """
        # Obtain numpy arrays
        _total_segmentation_array = sitk.GetArrayFromImage(self.total_segmentation_volume)
        _appendicular_segmentation_array = sitk.GetArrayFromImage(self.appendicular_segmentation_volume)
        _appendicular_segmentation_array = rearrange(_appendicular_segmentation_array, 'z y x -> x y z')        
        _total_segmentation_array = rearrange(_total_segmentation_array, 'z y x -> x y z')        

        # Do a label shift for appendicular
        _appendicular_segmentation_array += 117
        # Find which labels are present
        _total_segmentation_labels = np.unique(_total_segmentation_array)
        _appendicular_segmentation_labels = np.unique(_appendicular_segmentation_array)

        # Remove overlaping label
        _appendicular_segmentation_labels = np.delete(_appendicular_segmentation_labels, 
                                                      np.where(_appendicular_segmentation_labels == 117))
       
        # Obtain blobs
        ## Structure element for seeking blobs
        _storage_dict = {}
        for _i, _key in enumerate(self.remap_dict):
            #print(f"{_i+1}/{len(list(self.remap_dict.keys()))}")
            # Create binary array with 1 where the value is present
            if _key in _total_segmentation_labels:
                _binary_masked_array = (_total_segmentation_array == _key).astype(int)
            else:
                if _key in _appendicular_segmentation_labels:
                    _binary_masked_array = (_appendicular_segmentation_array == _key).astype(int)
                else:
                    continue
    
            # Obtain all disjoined blob
            _blob = np.where(_binary_masked_array == 1)
            _blob = list(zip(_blob[0], _blob[1], _blob[2]))
            
            # Convert everything to int
            _blob = [tuple(map(int, _tup)) for _tup in _blob]
            # Store it
            _storage_dict[_key] = _blob

        # Export it to json
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        if export_path != None:
            with open(os.path.join(export_path, f"{_name}_raw_blobs.json"), "w") as _outfile:
                json.dump(_storage_dict, _outfile)

        # Create variable
        self.raw_blobs = _storage_dict
        self.main_blobs_storage = _storage_dict

    def merge_blobs(self, 
                    merging_criteria: str = "reduced_cluster_remaped", 
                    json_path: str = None,
                    export_path: str = None):
        """
        Method which merges labels and blob into regions. It can also load json file (typically raw_blobs.json), but if not, then it uses
        dict stored in self.raw_blobs

        Args:
            json_path, str, if provided, path to the json dict if presented.
            merging_criteria, str, type of merging to be done
            export_path, str, if provided, path to where "merged_blobs.json" will be created
        """

        # Load if necessary
        if json_path != None:
            with open(json_path) as _json_file:
                self.main_blobs_storage = json.load(_json_file)
        
        # Populate raw_blobs by calculating raw if nothing is present
        if self.main_blobs_storage == None:
            self.obtain_raw_blobs()

        # Obtain named_dict:
        try:
            _df = pd.read_excel(self.cheatsheet_names, sheet_name = merging_criteria)
            _name_dict = _df.set_index("Label")["Name"].to_dict()
        except:
            print(f"Could not import names, aborting!!! Check if cheatsheet_names.xlsx exists")
            return -1
        
        # Merge based on criteria
        ## Obtain index
        self.merging_critera = merging_criteria
        if merging_criteria == "reduced_cluster_remaped":
            _index = 2
        if merging_criteria == "cluster_remaped":
            _index = 0
        if merging_criteria == "remaped":
            _index = 1

        # Merge
        _merged_blob_dict = {}
        for _key in self.main_blobs_storage:
            # Obtain label based on remaped dict
            _new_key = int(self.remap_dict[int(_key)][_index])
            if _new_key in _merged_blob_dict:
                _merged_blob_dict[_new_key] += self.main_blobs_storage[_key]
            else:
                _merged_blob_dict[_new_key] = self.main_blobs_storage[_key]

        # Change naming
        _tmp_dict = {}
        for _key in _merged_blob_dict:
            _new_key = _name_dict[_key]
            _tmp_dict[_name_dict[_key]] = _merged_blob_dict[_key]
        _merged_blob_dict = _tmp_dict

        # Export it to json
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        if export_path != None:
            with open(os.path.join(export_path, f"{_name}_merged_blobs.json"), "w") as _outfile:
                json.dump(_merged_blob_dict, _outfile)

        # Create dict
        self.merged_blobs = _merged_blob_dict
        self.main_blobs_storage = _merged_blob_dict

    def find_biggest_blobs(self,
                          json_path: str = None,
                          export_path: str = None):
        """
        Method which finds biggest blobs and store them into self.raw_blob variable. Also it
        exports them to bigest_blobs.json and can load raw.json or merged.json from json_path variable

        Args:
            * json_path, str, if provided, path to the json dict if presented.
            * export_path, str, if provided, path to where "merged_blobs.json" will be created
        """
        # Load if necessary
        if json_path != None:
            with open(json_path) as _json_file:
                self.main_blobs_storage = json.load(_json_file)
       
        # Obtain if none
        if self.main_blobs_storage == None:
            self.obtain_raw_blobs()

        # Format
        ## Structure element for seeking blobs
        _structure = [[[1 for _ in range(3)] for _ in range(3)] for _ in range(3)]
        _storage_dict = {}
        _size = self.total_segmentation_volume.GetSize()
        _size = (_size[0], _size[1], _size[2])
        for _i, _key in enumerate(self.main_blobs_storage):
            #print(f"{_i+1}/{len(list(self.main_blobs_storage.keys()))}")
            # Create binary array with 1 where the value is present
            _binary_masked_array = np.zeros(_size)
            _x, _y, _z = zip(*self.main_blobs_storage[_key])
            _x = np.array(_x)
            _y = np.array(_y)
            _z = np.array(_z)
            _binary_masked_array[_x, _y, _z] = 1 
            _binary_masked_array =_binary_masked_array.astype(int)

            # Obtain blobs
            _labeled_array, _num_features = label(input = _binary_masked_array, 
                                                  structure = _structure)
            # Calculate blobs
            _blobs = []
            for _feature_num in range(1, _num_features + 1):
                _blob = np.where(_labeled_array == _feature_num)
                _blobs.append(list(zip(_blob[0], _blob[1], _blob[2])))
        
            # Obtain the biggest blob
            if _blobs != []:
                _biggest_blob_index = [len(_blob) for _blob in _blobs]
                _biggest_blob_index = _biggest_blob_index.index(max(_biggest_blob_index))
                _biggest_blob = _blobs[_biggest_blob_index]

                # Convert everything to int
                _biggest_blob = [tuple(map(int, _tup)) for _tup in _biggest_blob]
                # Store it
                _storage_dict[_key] = list(_biggest_blob)
            
        # Export it to json
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        if export_path != None:
            with open(os.path.join(export_path, f"{_name}_biggest_blobs.json"), "w") as _outfile:
                json.dump(_storage_dict, _outfile)

        # Create variable
        self.biggest_blobs = _storage_dict
        self.main_blobs_storage = _storage_dict
         
    def filter_blobs(self, json_path: str = None, xlsx_path: str = None, export_path: str = None):
        """
        Method which applies filtering on the blobs in memory based on the xlsx_path provided. Also it can load any
        dict and perfrome filtering on it.
        
        Args:
            * json_path, str, if provided, path to the json dict if presented.
            * xlsx_path, str, path to the xlsx file where the areas are filtered and storend under col name "True"
            * export_path, str, if provided, path to where input xlsx will be saved as filtered_data.xlsx

        """
        # Read labels
        try:
            _df = pd.read_excel(xlsx_path, index_col = 0 )
            _true_labels = _df.loc["True"]
        except:
            print(f"Error while reading provided xlsx!. Column name 'True' migh be missing!")
            return -1
        
        # Generate blobs
        # Load if necessary
        if json_path != None:
            with open(json_path) as _json_file:
                self.main_blobs_storage = json.load(_json_file)

        # Obtain blobs
        if self.main_blobs_storage == None:
            self.obtain_raw_blobs()
        
        # Delete key if necessary
        _keys = list(self.main_blobs_storage.keys())
        for _key in _keys:
            if _true_labels[_key] == 0:
                del self.main_blobs_storage[_key]

        # Save the xlsx path in the input dir
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        _df.to_excel(os.path.join(export_path, f"{_name}_filtered_data.xlsx"))


        if export_path != None:
            with open(os.path.join(export_path, f"{_name}_filtered_blobs.json"), "w") as _outfile:
                json.dump(self.main_blobs_storage, _outfile)

        # Store, main_blobs_storage is already updated
        self.filtered_blobs = self.main_blobs_storage
        
    def generate_nifti(self, json_path: str = None, export_path:str = None, file_name: str = None):
        """
        Method which exports merged labels to nifti file in the given export path

        Args:
            * json_path, str, if provided, path to the json dict if presented.
            * export_path, str, path to the export dir.
            * file_name, str, if file name = None, then file name is set to OUTPUT_SEG.nii
        """
        # Storage
        self.main_segmentation_volume = None

        # Test for all necessary variables
        assert export_path!=None, f"Missing export path, please define one!"
        assert self.total_segmentation_volume != None and self.appendicular_segmentation_volume != None, \
            f"Missing volumes! Appendicular and total volume missing!"
        
        # Obtain all necessary data for nifti file recreation
        if self.total_segmentation_volume == None:
            _volume = self.appendicular_segmentation_volume
        else:
            _volume = self.total_segmentation_volume

        # Obtain necessary data for export
        ## Rearange it and transfer it to sitk.Image
        _size = _volume.GetSize()
        _origin = _volume.GetOrigin()
        _direction = _volume.GetDirection()
        _spacing = _volume.GetSpacing()
        
        ## Get names to generate values
        if self.main_blobs_storage == None or self.merging_critera == None:
            self.obtain_raw_blobs()
        _df = pd.read_excel(self.cheatsheet_names, sheet_name = self.merging_critera)
        _name_dict = _df.set_index("Name")["Label"].to_dict()
        
        # Generate image
        _img = np.zeros(_size)
        for _key in self.main_blobs_storage:
            _points_array = np.array(self.main_blobs_storage[_key])
            _img[_points_array[:, 0], _points_array[:, 1], _points_array[:, 2]] = _name_dict[_key]

        _img = rearrange(_img, 'x y z -> z y x')
        _img = sitk.GetImageFromArray(_img)
        

        # Load image properties to the image from the dict
        _img.SetOrigin(_origin)
        _img.SetSpacing(_spacing)
        _img.SetDirection(_direction)

        # Create main segmentation volume
        self.main_segmentation_volume = _img

        # Export
        if file_name == None:
            _file_name = "OUTPUT_SEG.nii"
        else:
            _file_name = file_name
        _name = os.path.basename(self.total_segmentation_path).split("_Total")[0]
        sitk.WriteImage(_img, os.path.join(export_path, f"{_name}_{_file_name}"))

    def __read_main_volumes(self):
        """
        Method which reads main nifti file.
        """
        # Load volumes
        self.main_ct_volume = None
        self.main_segmentation_volume = None
        self.appendicular_segmentation_volume = None
        self.main_ct_volume = sitk.ReadImage(self.main_ct_path)
        self.total_segmentation_volume = sitk.ReadImage(self.total_segmentation_path)
        self.appendicular_segmentation_volume = sitk.ReadImage(self.appendicular_segmentation_path)

    def __obtain_data(self):
        """
        Method which detects important files in input direcotry and stores them into
        appropriate variables.
        """
        # Place holders
        self.json_file_path = None
        self.main_ct_path = None
        self.main_image_path = None
        self.total_segmentation_path = None
        self.appendicular_segmentation_path = None
        # Obtain list of files
        for _, _, _files in os.walk(self.input_dir):
            for _file in _files:
                # Assign file paths to the vairables
                # JSON
                if os.path.splitext(_file)[-1] == ".json":
                    self.json_file_path = os.path.join(self.input_dir, _file)      

                # NII main
                if ".nii.gz" in _file:
                    self.main_ct_path = os.path.join(self.input_dir, _file)
                
                # NII appendicular
                if "appendicular_bones" in _file:
                    self.appendicular_segmentation_path = os.path.join(self.input_dir, _file)
                        
                # NII appendicular
                if "total" in _file:
                    self.total_segmentation_path = os.path.join(self.input_dir, _file)
                
                # Image
                if "_reducted_image" in _file:
                    self.main_image_path = os.path.join(self.input_dir, _file)
        
        # Check for all files
        #assert os.path.exists(self.json_file_path), f"Missing file: {self.json_file_path}!!! Aborting!"
        assert os.path.exists(self.main_ct_path), f"Missing file: {self.main_ct_path}!!! Aborting!"
        assert os.path.exists(self.main_image_path), f"Missing file: {self.main_image_path}!!! Aborting!"
        assert os.path.exists(self.total_segmentation_path), f"Missing file: {self.total_segmentation_path}!!! Aborting!"
        assert os.path.exists(self.appendicular_segmentation_path), f"Missing file: {self.appendicular_segmentation_path}!!! Aborting!"

    def __get_color_palette(self, palette_name: str ='tab20c', num_colors: int = 10)->list:
        """
        Method for colormap, quite simple but effective

        Args:
            * pallete_name, str, name of the color pallete which will be applied
            * num_color, int, number of color in the pallete

        Returns:
            list, list of colour codes.
        """
        _cmap = plt.get_cmap(palette_name)
        # Exclude the alpha channel by taking only the first three elements (R, G, B)
        colors = [_cmap(_i)[:3] for _i in np.linspace(0, 1, num_colors)]
        return colors

    def __export_to_2d(self, 
                       volume: sitk.Image = None,
                       export_name: str = "default.png",
                       series_info_dict: dict = None,
                       new_spacing: tuple = (0.5, 0.5, 0.5), 
                       lower_bound: int = -1024,
                       upper_bound: int = 1500,
                       histogram_reduction_params: dict = {"search_region": 400,
                                                        "reduction_factor": 50,
                                                        "number_of_bins": 500,
                                                        "upper_bound": 1500}):
        """
        Method which transfer volumes to the 2D images

        Args:
            * volume, sitk.img, volume to be transfered
            * export_name, str, name under which the volume will be saved
            * series_info_dict, dict, dictionary which contains projection of the image.
            * new_spacing, tuple, tuple of integers representing spacing for the simpleitk image
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
                for more info. The selected params were the best one.
                        * lower_bound, int, inital corropping value for the volume. Everything < lower_bound = lower_bound
            * upper_bound, int, initial cropping value for the volume. Everything > upper_bound = upper_bound
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
            for more info. The selected params were the best one.

        """
         # Resample image
        _resampled_image = resample_image(image = volume,
                                            new_spacing = new_spacing)
        # Go to numpy
        _resampled_item = transfer_to_numpy(input_image = _resampled_image)
        # Bound the volume
        _resampled_item['image'] = bound_volume(volume = _resampled_item['image'],
                                                lower_bound = lower_bound,
                                                upper_bound = upper_bound)
         # Export coronal image
        _image = transfer_to_coronal(image = _resampled_item['image'], 
                    tags = series_info_dict["series_tags"])

        # Do histogram reduction
        if "lower" in list(histogram_reduction_params.keys()):
            _min_value = histogram_reduction_params["min"]
            _reducted_image = np.where(_image < histogram_reduction_params["lower"], _min_value, _image)
            _reducted_image = np.where(_reducted_image > histogram_reduction_params["upper"], upper_bound, _reducted_image)
        else :
            _reducted_image, _histogram_reduction_params, _min = histogram_reduction(image = _image, 
                                                        search_region = histogram_reduction_params['search_region'],
                                                        reduction_factor = histogram_reduction_params['reduction_factor'],
                                                        number_of_bins = histogram_reduction_params['number_of_bins'],
                                                        upper_bound = histogram_reduction_params['upper_bound'])
        
        # Export reducted image
        _summed_2d_array = np.sum(_reducted_image, axis=2)
        #Apply colormap gray from matplotlib
        if "array_min" in list(histogram_reduction_params.keys()):
            _norm = plt.Normalize(vmin = histogram_reduction_params["array_min"], 
                                  vmax = histogram_reduction_params["array_max"])
        else:
            _norm = plt.Normalize(vmin = _summed_2d_array.min(), 
                                vmax = _summed_2d_array.max())
        
        _image_norm = plt.cm.gray(_norm(_summed_2d_array))
        _image_norm = (_image_norm[:,:, 0] * 255).astype(np.uint8)
        _image_norm = Image.fromarray(_image_norm.T)
        _image_norm.save(export_name)
        # Return params
        if "search_region" in list(histogram_reduction_params.keys()):
            return {"lower": _histogram_reduction_params[0], 
                    "upper": _histogram_reduction_params[1], 
                    "min": _min,
                     "array_min": _summed_2d_array.min(),
                     "array_max": _summed_2d_array.max()}

    

    def __export_labels_to_2d(self,
                              volume: sitk.Image = None,
                              series_info_dict: dict = None,
                              new_spacing: tuple = (0.5, 0.5, 0.5),
                              export_path:str = None,
                              merging_criteria: str = "reduced_cluster_remaped",
                              tolernace: int = 10):
        """
        Method which extract images of detected regions from the given patch region.
        
        Args:
            * volume, sitk.img, volume to be transfered
            * series_info_dict, dict, dictionary which contains projection of the image.
            * new_spacing, tuple, tuple of integers representing spacing for the simpleitk image
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
            for more info. The selected params were the best one.
            * export_path, str, path to place where the images/json will be stored.
            * merging_criteria, str, type of merging to be done
            * tolerance, int, if label has sub 10 pixels- it is dropped and considered as 0

        """
        # Obtain named_dict:
        try:
            _df = pd.read_excel(self.cheatsheet_names, sheet_name = merging_criteria)
            _name_dict = _df.set_index("Label")["Name"].to_dict()
            _output_dict = _df.set_index("Name")["Label"].to_dict()
        except:
            print(f"Could not import names, aborting!!! Check if cheatsheet_names.xlsx exists")
            return -1
        
        # obtain colorpalette
        _color_pallete = self.__get_color_palette(num_colors=len(list(_name_dict.keys()))+1)
        # Get array
        _volume_array = sitk.GetArrayFromImage(volume)
        _volume_array = rearrange(_volume_array, 'z y x -> x y z')

        # Get labels
        _unique_values, _counts = np.unique(_volume_array, return_counts=True)
        # remove 0
        _unique_values = _unique_values[1:]
        _counts = _counts[1:]
        _labels_in_volume = dict(zip(_unique_values, _counts))       
        
        for _key in _labels_in_volume:
            # Get array
            _mask_indexes = _volume_array == _key
            _mask_volume_array = np.zeros(_volume_array.shape)
            _mask_volume_array[_mask_indexes] = _key

            # Get volume
            _mask_volume_array = rearrange(_mask_volume_array, 'x y z -> z y x')
            _mask_volume = sitk.GetImageFromArray(_mask_volume_array)
            _mask_volume.SetOrigin(volume.GetOrigin())
            _mask_volume.SetSpacing(volume.GetSpacing())
            _mask_volume.SetDirection(volume.GetDirection())          
            
            # Resample
            _resampled_image = resample_image(image = _mask_volume,
                                            new_spacing = new_spacing)

            # Go to numpy
            _resampled_item = transfer_to_numpy(input_image = _resampled_image)

            # Export coronal image
            _image = transfer_to_coronal(image = _resampled_item['image'], 
                    tags = series_info_dict["series_tags"])
            
            # Obtain 2d image
            _mask = np.any(_image == _key, axis = 2)
            _mask = _mask.T
            _mask = np.round(_mask)
            _mask = (_mask != 0).astype(np.uint8)

            # Update labels
            _labels_in_volume[_key] = np.sum(_mask)
            if _labels_in_volume[_key] < 10:
                _labels_in_volume[_key] = 0
                continue
            _rgb_image = np.zeros((_mask.shape[0], _mask.shape[1], 3))
            _rgb_image[_mask != 0] = _color_pallete[int(_key)]

            # Export
            _name = _name_dict[_key]
            _img = Image.fromarray((_rgb_image * 255).astype(np.uint8))
            _name = os.path.join(export_path, f"{_name}.png")
            _img.save(_name)

        # Output xlsx with labels
        # Reset areas
        for _key in _output_dict:
            _output_dict[_key] = 0
        for _key in _labels_in_volume:
            _output_dict[_name_dict[_key]] = _labels_in_volume[_key]

        _df = pd.DataFrame(data=[_output_dict.values()], columns=_output_dict.keys())
    
        # Add a first column with "Name" and "Value"
        _df.insert(0, "Name", ["Value"])
        # Write the DataFrame to an Excel file
        _df.to_excel(os.path.join(export_path, f"labels.xlsx"), index=False, header=True)

    def __generate_main_validation_image(self, folder:str = None, min_area:int = 1):
        """
        Method which generates validation image of a case.
        
        Args:
            * folder, str, path to export folder
            * min_area, int, min % of area in all images to be present 
        """
        # Obtain areas
        _df = pd.read_excel(os.path.join(folder, "labels.xlsx"), header= None, index_col=0)

        # Extract the first row (keys) and second row (values)
        _keys = _df.iloc[0]
        _values = _df.iloc[1]
        
        # Create a dictionary from the keys and values
        _data_dict = dict(zip(_keys, _values))

        # Storage
        _images = []
        _filenames = []

        # Valid
        _valid = True
        # Filename obtaining
        for _filename in sorted(os.listdir(folder)):
            if _filename.endswith(".png"):
                _img = cv2.imread(os.path.join(folder, _filename))
                if _img is not None:
                    if _filename != "main.png":
                        _images.append(_img)
                        _filenames.append(_filename.split(".png")[0])
                    else:
                        _main_img = _img
                        _main_filename = _filename.split(".png")[0]
        # Add main
        _images.insert(0, _main_img)
        _filenames.insert(0, _main_filename)
        
        # Create image to display
        _n = len(_images)
        _grid_size = math.ceil(math.sqrt(_n))
        
        # Create plot
        _fig, _axes = plt.subplots(_grid_size, _grid_size, figsize=(15, 15))
        _axes = _axes.flatten()
        
        # Plot it
        for _ax in _axes:
            _ax.axis('off')
            
        for _i, (_img, _filename) in enumerate(zip(_images, _filenames)):
            _axes[_i].imshow(_img)
            if _filename != "main":
                # Check for validity:
                if _data_dict[_filename]/(_img.shape[0]*_img.shape[1])*100 < min_area:
                    _valid = False
                _axes[_i].set_title(f"{_filename}: {_data_dict[_filename]/(_img.shape[0]*_img.shape[1])*100:4.1f}%", 
                                    fontsize = 24)
            else:
                _axes[_i].set_title(f"{_filename}: {_img.shape[0]*_img.shape[1]}", fontsize = 24)
            _axes[_i].axis('off')

        plt.tight_layout()
        
        # Save
        _fig.savefig(os.path.join(folder, "all.png"))
        #plt.cla()
        plt.close(_fig)

        # Return validity
        return _valid
    
    def transfer_coronal_to_original(self, image: np.array = None, 
                       tags: dict = None)->np.array:
        """
        Function which generate original image projection from coronal.
        Args:
            * image, np.array, numpy array containing image 
            * tags, dict, dict containing 'series_tags'

        Returns:
            * image, np.array, oriented and transposed image in case the image is in
            saggittal, axial or coronal projection. Othwervise, return the same
            image.
        """
        # Set rotation keys
        _coronal_plane = np.array([1, 0, 0, 0, 0, -1])  
        _saggittal_plane = np.array([0, 1, 0, 0, 0, -1]) 
        _axial_plane =np.array([1, 0, 0, 0, 1, 0])
        
        # Parse rotation
        try:
            _tag_value = tags["0020|0037"][0]
            _tag_value = np.array(_tag_value.split('\\')).astype(float)
            _tag_value = np.rint(_tag_value).astype(int)
        except:
            print(f"Missing tag value, could not obtain tag ImageOrientationPatient--0020|0037 in provided dict, returning same image")
            return image
        # Coronal -- no change
        if np.array_equal(_tag_value, _coronal_plane): 
            return image
        
        # Saggittal plane
        if np.array_equal(_tag_value, _saggittal_plane):
            image = rearrange(image, 'z y x -> x y z')
            return image

        # Axial plane TO BE TESTED 
        if np.array_equal(_tag_value, _axial_plane):
            image = image[:, ::-1, :]
            image = rearrange(image, 'x z y -> x y z')
            return image
        
        # Not found any matchin plane -> returning the same image
        return image
    
    def list_to_coronal(self, input: list = None, 
                           tags: dict = None,
                           reverse: int = None)->list:
        """
        Method which rearrange indexes of list into coronal order

        Args:
            * input, list, list with typically 3 elements
            * tags, dict, dict containing 'series_tags'
            * reverse, int, necessary for axial


        Return:
            * rearanged list of three elements in coronal shape
        """
         # Set rotation keys
        _coronal_plane = np.array([1, 0, 0, 0, 0, -1])  
        _saggittal_plane = np.array([0, 1, 0, 0, 0, -1]) 
        _axial_plane =np.array([1, 0, 0, 0, 1, 0])
        
        # Parse rotation
        try:
            _tag_value = tags["0020|0037"][0]
            _tag_value = np.array(_tag_value.split('\\')).astype(float)
            _tag_value = np.rint(_tag_value).astype(int)
        except:
            print(f"Missing tag value, could not obtain tag ImageOrientationPatient--0020|0037 in provided dict, returning same image")
            return input
        # Coronal -- no change
        if np.array_equal(_tag_value, _coronal_plane): 
            return input
        
        # Saggittal plane
        if np.array_equal(_tag_value, _saggittal_plane):
            input[0], input[2] = input[2], input[0]
            return input

        # Axial plane TO BE TESTED 
        if np.array_equal(_tag_value, _axial_plane):
            input[1], input[2] = input[2], input[1]
            if reverse != None:
                input[1] = reverse - input[1] - 1 
            return input
        
        # Not found any matchin plane -> returning the same image
        return input
    
    def create_central_patches(self,
                               series_info_path: str = None, 
                               biggest_blobs_path: str = None,
                               segmentation_volume_path: str = None,
                               new_spacing: tuple = (0.5, 0.5, 0.5),
                               merging_criteria: str = "reduced_cluster_remaped",
                               lower_bound: int = -1024,
                               upper_bound: int = 1500,
                               histogram_reduction_params: dict = {"search_region": 400,
                                                        "reduction_factor": 50,
                                                        "number_of_bins": 500,
                                                        "upper_bound": 1500},
                               width_height_dict: dict = {"min": 128, "max": 400, "n": 10},
                               min_area = 0.5):
        """
        Method which creates central patches and stores them in given directory.

        Args:
            * biggest_blobs_path, str, either nii or json file leading to the biggest blobs
            * sgementation_volum_path, str, path to nii segmentation map - output of total segmentor refined by labels
            * spacing, tuple, tuple of integers representing spacing for the simpleitk image
            * merging_criteria, str, type of merging to be done

        """
        # Check for necessary info
        _series_info_dict = None
        _file = None
        if series_info_path != None:
            _file = series_info_path
        else:
            _root, _ext = os.path.splitext(self.main_ct_path)
            _root, _ext = os.path.splitext(_root)

            # Change the extension to .json
            _file = _root + '.json'
        assert _file != None, f"Could not access series_info_path file, please check the provided src"

        # Load json
        _file = open(_file)    
        _series_info_dict = json.load(_file)

        # Get biggest blobs
        _biggest_blobs_dict = None

        # Load necessary data if not defined
        if biggest_blobs_path != None:
            _file_extension = os.path.splitext(biggest_blobs_path)[-1]
            # JSON
            if _file_extension == ".json":
                _f = open(biggest_blobs_path)
                _biggest_blobs_dict = json.load(_f)
            
            # NII
            if _file_extension == ".nii":
                # Obtain named_dict:
                try:
                    _df = pd.read_excel(self.cheatsheet_names, sheet_name = merging_criteria)
                    _name_dict = _df.set_index("Label")["Name"].to_dict()
                except:
                    print(f"Could not import names, aborting!!! Check if cheatsheet_names.xlsx exists")
                    return -1

                # Obtain image
                _sitk_image = sitk.ReadImage(biggest_blobs_path)
                _blobs_arrays = sitk.GetArrayFromImage(_sitk_image)
                _blobs_arrays = rearrange(_blobs_arrays, 'z y x -> x y z')   
                _unique_values = np.unique(_blobs_arrays)
                _mask = _unique_values != 0
                _unique_values = _unique_values[_mask]
                _coordinates = np.argwhere(_blobs_arrays >= -np.inf)

                # Initialize the dictionary
                _biggest_blobs_dict = {_val: [] for _val in _unique_values}

                # Populate the dictionary with coordinates
                for _val in _unique_values:
                    _biggest_blobs_dict[_val] = _coordinates[_blobs_arrays[tuple(_coordinates.T)] == _val]
                
                # Swap names
                _keys = list(_biggest_blobs_dict.keys())
                for _key in _keys:
                    _biggest_blobs_dict[_name_dict[int(_key)]] = _biggest_blobs_dict.pop(_key)
                    _biggest_blobs_dict[_name_dict[int(_key)]] = _biggest_blobs_dict[_name_dict[int(_key)]].tolist()
        # Obtain from container
        if _biggest_blobs_dict == None and self.biggest_blobs != None:
            _biggest_blobs_dict = self.biggest_blobs
    
        # Final check
        assert _biggest_blobs_dict != None, f"Problem with finding the biggest blob, please check paths and provide biggest bolbs"
            
        # Obtain segmentation volume
        _segmentation_volume_array = None
        if segmentation_volume_path != None:
            _segmentation_volume = sitk.ReadImage(segmentation_volume_path)
            _segmentation_volume_array = sitk.GetArrayFromImage(_segmentation_volume)
            _segmentation_volume_array = rearrange(_segmentation_volume_array, 'z y x -> x y z')
            _segmentation_volume_array = transfer_to_coronal(image =_segmentation_volume_array,
                                                            tags = _series_info_dict["series_tags"])

               
        # Final check
        assert _segmentation_volume_array is not None, f"Problem with finding the segmentation volume, please check paths and provide segmentation volume"
        
        # Obtain center points
        _center_points_dict = {}
        # Calculate centers and add them
        for _key in _biggest_blobs_dict:
            _center_point = np.array(_biggest_blobs_dict[_key]).mean(axis = 0)
            _center_point = [int(_center_point[0]), int(_center_point[1]), int(_center_point[2])]
            # Store it
            _center_points_dict[_key] = _center_point
                    
        # Main volume
        _main_volume = sitk.ReadImage(self.main_ct_path)
        _main_volume_array = sitk.GetArrayFromImage(_main_volume)
        _main_volume_array = rearrange(_main_volume_array, 'z y x -> x y z')
        _main_volume_array = transfer_to_coronal(image = _main_volume_array, 
                    tags = _series_info_dict["series_tags"])

        # Change ratios if necessary


        # Export and obtain params for segmentation export
        _histogram_reduction_params_patches = self.__export_to_2d(volume = _main_volume,
                            export_name= "main.png",
                            series_info_dict=_series_info_dict,
                            new_spacing = new_spacing,
                            lower_bound= lower_bound,
                            upper_bound=upper_bound,
                            histogram_reduction_params = histogram_reduction_params)
        
        # Make storage
        _dir = os.path.join(self.input_dir, "images")
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        os.makedirs(_dir)

        # get centre crop points
        ## Obtain list of width, height shapes
        _original_spacing = _segmentation_volume.GetSpacing()
        _new_spacing = new_spacing
        _ratios = np.array(_original_spacing) / np.array(_new_spacing)
        
        # Transfer ratios if necessary
        _ratios = self.list_to_coronal(input = _ratios.tolist(),
                                       tags = _series_info_dict["series_tags"])

        for _key in _center_points_dict:
            for _n in range(width_height_dict["n"]):
                # Obtain random width and height
                _width, _height = np.random.randint(width_height_dict["min"], 
                                                    width_height_dict["max"] + 1, 2)
                
                _width = int(_width // _ratios[0])
                _height = int(_height // _ratios[1])
                _volume_x, _volume_y, _volume_z = _segmentation_volume_array.shape

                # Obtain center point cordinates
                _x, _y, _z = _center_points_dict[_key]
                _x, _y, _z = self.list_to_coronal(input = [_x, _y, _z],
                                                    tags = _series_info_dict["series_tags"],
                                                    reverse = _volume_y)

                # Calculate half width and half height
                _half_width = _width // 2
                _half_height = _height // 2

                _start_x = _x - _half_width
                _end_x = _x + _half_width + (_width % 2)

                _start_y = _y - _half_height
                _end_y = _y + _half_height + (_height % 2)

                #_start_z = _z - _half_height
                #_end_z = _z + _half_height + (_height % 2)
                
                if _start_y < 0:
                    _end_y += abs(_start_y)
                    _start_y = 0
                if _end_y > _volume_y:
                    _start_y -= (_end_y - _volume_y)
                    if _start_y < 0:
                        _start_y = 0
                    _end_y = _volume_y

                # Adjust if out of bounds
                if _start_x < 0:
                    _end_x += abs(_start_x)
                    _start_x = 0
                if _end_x > _volume_x:
                    _start_x -= (_end_x - _volume_x)
                    if _start_x < 0:
                        _start_x = 0
                    _end_x = _volume_x           
                #if _start_z < 0:
                #    _end_z += abs(_start_z)
                #    _start_z = 0
                #if _end_z > _volume_z:
                #    _start_z -= (_end_z - _volume_z)
                #    _end_z = _volume_z

                # Finally segment a volume
                #_segmented_volume_patch = _segmentation_volume_array[_start_x:_end_x, :, _start_z:_end_z]
                #_main_volume_patch = _main_volume_array[_start_x:_end_x, :, _start_z:_end_z] = 255
                _segmented_volume_patch = _segmentation_volume_array[_start_x:_end_x, _start_y: _end_y, :]
                _main_volume_patch = _main_volume_array[_start_x:_end_x, _start_y: _end_y, : ]
                
                # Main_volume_patch
                _main_volume_patch = self.transfer_coronal_to_original(image=_main_volume_patch,
                                                                        tags = _series_info_dict["series_tags"])
                _main_volume_patch = rearrange(_main_volume_patch, 'x y z -> z y x')
                _main_volume_patch = sitk.GetImageFromArray(_main_volume_patch)

                _main_volume_patch.SetOrigin(_main_volume.GetOrigin())
                _main_volume_patch.SetSpacing(_main_volume.GetSpacing())
                _main_volume_patch.SetDirection(_main_volume.GetDirection())
                
                # Segment_volume_patch
                _segmented_volume_patch = self.transfer_coronal_to_original(image = _segmented_volume_patch,
                                                                            tags = _series_info_dict["series_tags"])
                _segmented_volume_patch = rearrange(_segmented_volume_patch, 'x y z -> z y x')
                _segmented_volume_patch = sitk.GetImageFromArray(_segmented_volume_patch)
                _segmented_volume_patch.SetOrigin(_main_volume.GetOrigin())
                _segmented_volume_patch.SetSpacing(_main_volume.GetSpacing())
                _segmented_volume_patch.SetDirection(_main_volume.GetDirection())

                # Create export dir 
                _name = f"{_key}_{_n}"
                _patch_dir = os.path.join(_dir, _name)
                os.makedirs(_patch_dir)

                self.__export_to_2d(volume=_main_volume_patch,
                                    series_info_dict=_series_info_dict,
                                    histogram_reduction_params = _histogram_reduction_params_patches,
                                    export_name= os.path.join(_patch_dir, "main.png"))
                                
                self.__export_labels_to_2d(volume = _segmented_volume_patch,
                                            series_info_dict=_series_info_dict,
                                            new_spacing = new_spacing,
                                            merging_criteria = "reduced_cluster_remaped",
                                            export_path = _patch_dir)

                _valid = self.__generate_main_validation_image(folder = _patch_dir, min_area = min_area)

                # Check validity
                # Area problem
                if _valid == False:
                    shutil.rmtree(_patch_dir)
            
