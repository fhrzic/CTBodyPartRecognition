import numpy as np
import os
import SimpleITK as sitk
from einops import rearrange
import json
import zipfile
import shutil
from Utils.Conversion import *
from PIL import Image           
from collections import Counter

# Utility Functions
#################################################

def extract_zip(input_zip_path: str = None, 
                target_dir_path: str = ".", 
                case_number: str = None,
                preserve_dir_structure = True,
                verbose = True)->str:
    """
    Function which extracts all series(dcm) from input_zip_path to target_dir. 
    Series case number must be provided either in zip file name or manually (manuall overrides
    extracted one if provided). Path is "target_dir_path/case_number_exp"

    Args:
        * input_zip_path, str, path to the zip which should be extracted
        * target_dir_path, str, path to where the data will be extracte - parent dir
        * case_number, str, custom name if necessary - in case that zip file
        does not have case number in it.
        * preserve_dir_structure, bool, if True, the dir sturcture of the zip will
        be preserved, if False, all ".dcm" will be extracted to one dir. CAUTION:
        SERIES WORKS MUCH FASTER IF THE DIR STRUCTURE IS PRESERVED (x8)
        * verbose, bool, if True, the info about progress will be reported
    
    Returns:
        *export_dir_path, str, path where the zip file is exported.
    """
    assert input_zip_path != None or target_dir_path != None, \
        f"Missing data input_zip_path must be provided and target_dir_path must not be None"
    
    # Extract case number
    _case_number = None
    _case_number = [_s for _s in input_zip_path.split(os.sep) if "case" in _s][0]
    _case_number = os.path.splitext(_case_number)[0]

    if case_number != None:
        _case_number = case_number
    assert _case_number != None, \
        f"ERROR: case_number must not be None!!!(could not extract number. please provide one!)"
    
    # Generate export dir path
    _export_dir_path = os.path.join(target_dir_path, f"{_case_number}_exp")

    # Check if directory exists -> create if not
    if not os.path.exists(_export_dir_path):
        os.makedirs(_export_dir_path)
        if verbose:
            print(f"Directory {format(_export_dir_path)} created.")
    else:
        if verbose:
            print(f"Directory {format(_export_dir_path)} already exists, appending to it!")

    # Extract whole zip preserving folder structure
    if preserve_dir_structure:
        with zipfile.ZipFile(input_zip_path) as _zip_file:
            _zip_file.extractall(_export_dir_path)
    else:
        # Extract all ".dcm" in zip to one folder
        with zipfile.ZipFile(input_zip_path) as _zip_file:
            for _member in _zip_file.namelist():
                _filename = os.path.basename(_member)
                # Skip directories
                if not _filename:
                    continue
                if _member.endswith('.dcm'):
                    _middle_dir_name = os.path.basename(os.path.dirname(_member))
                    # copy file (taken from zipfile's extract)
                    _source = _zip_file.open(_member)
                    _target = open(os.path.join(_export_dir_path, f"{_middle_dir_name}_{_filename}"), "wb")
                    with _source, _target:
                        shutil.copyfileobj(_source, _target)
        
    # Returns the dir where the zip is extracted
    return _export_dir_path

def reconstruct_image(input_series_dict: dict = None, 
                      display: bool = False)->sitk.Image:
    """
    Function which returns sitk.Image extracted from given dictionary. 
    
    Args:
        * input_series_dict, dict, dictionary obtained in SeriesContainer for one series 
        including following keys: 'main_dir', 'series_ID', 'series_tags', 'series_image_info'
        'image'.
        * display, bool, if True, it will display the image in external tool provided in
        SITK_SHOW_COMMAND
    """
    # Check input
    assert input_series_dict != None and set(['main_dir', 'series_ID', 'series_tags', 'series_image_info'
        'image']) != set(input_series_dict.keys()), f"Keys missmatch, please provide dict obtained in class SeriesContainer"

    # Load image
    _img = input_series_dict['image']

    # Rearange it and transfer it to sitk.Image
    _img = rearrange(_img, 'x y z -> z y x')
    _img = sitk.GetImageFromArray(_img)

    # Load image properties to the image from the dict
    _img.SetOrigin(input_series_dict['series_image_info']['origin'])
    _img.SetSpacing(input_series_dict['series_image_info']['spacing'])
    _img.SetDirection(input_series_dict['series_image_info']['direction'])

    # If display is true, display it, return img
    if display:
        sitk.Show(_img)
    return _img

def resample_image(image: sitk.Image = None,
                   new_spacing: list = [1, 1, 1])->sitk.Image:
        """
        Function for resampling the image. 
        Args:
            * image, sitk.Image, image which will be resampled to new dimensions
            * new_spacing, list, list of the new spacing for the output image

        Returns:
            * resampled_image, sitk.Image, resampled sitk image
        """
        # Create resampling engine
        _resample_engine = sitk.ResampleImageFilter()
        _resample_engine.SetInterpolator(sitk.sitkLinear)
        _resample_engine.SetOutputDirection(image.GetDirection())
        _new_spacing = new_spacing
        _resample_engine.SetOutputOrigin(image.GetOrigin())
        _resample_engine.SetOutputSpacing(_new_spacing)

        # Calculate size
        _original_size = np.array(image.GetSize(), dtype = np.int64)
        _original_spacing = np.array(image.GetSpacing())
        _new_spacing = np.array(_new_spacing)
        _new_size = _original_size * (_original_spacing / _new_spacing)
        _new_size = np.ceil(_new_size).astype(np.int64)
        _resample_engine.SetSize(_new_size.tolist())

        
        # Resample image
        _resampled_image = _resample_engine.Execute(image)
        # Return 
        return _resampled_image

def transfer_to_numpy(input_image: sitk.Image)->dict:
    """
    Function which returns np.array version of the image and 
    dict which has all image attributs

    Args:
        * input_image, sitk.Image, input image which we want to transfer
        to np array

    Returns:
        * output_dict, dict, dictionary in following format
            * 'series_image_info': minimal requred image info for its reconstruction:
            size, spacing, origin, direction
            * 'image': image array with x y z oriented channels (solving sitk issue)
    """
    _image_info_dict = {'size': input_image.GetSize(),
                                    'spacing': input_image.GetSpacing(),
                                    'origin': input_image.GetOrigin(),
                                    'direction': input_image.GetDirection()
                        }
                
    # Transfering image to numpy array and reshaping it
    _image_numpy_array = sitk.GetArrayFromImage(input_image)                
    _image_numpy_array = rearrange(_image_numpy_array, 'z y x -> x y z')

    # Output dict:
    _output_dict = {'series_image_info': _image_info_dict,
                    'image': _image_numpy_array}
    # Return
    return _output_dict

# Directory structure
#################################################
"""
Expected file structure:
    main_dir:-> series_dir#1:
                        -> xxx_1.dcm
                        -> xxx_2.dcm
                        ...
                        -> xxx_n.dcm
            -> series_dir#2:
                        -> xxx_1.dcm
                        -> xxx_2.dcm
                        ...
                        -> xxx_n.dcm
            -> series_dir#n:

or
    main_dir:
    -> xxx_1.dcm
    -> xxx_2.dcm
    ...
    -> xxx_n.dcm
"""
#################################################

class SeriesContainer():
    """
    Class which loads series and stores them in a container. Container is a list of series propertis 
    which are sotred as dictionary with following keys
        * 'main_dir': series directory name (where the series is located)
        * 'series_ID': series ID
        * 'series_tags': defined in the __init_tags method, printable with print_tags method
        * 'series_image_info': minimal requred image info for its reconstruction:
        size, spacing, origin, direction
        * 'image': image array with x y z oriented channels (solving sitk issue)
    """

    def __init__(self, 
                 verbose: bool = True):
        """
        Init method.

        Args:
            * path_dir, str, default is None. Path to the directory containing possible series.
            If multiple directories are inside the main dir, then it extract those directories 
            as potential candidate for series.
            * verbose, bool, for debugging and printing auxiliary results
        """
        # Set verbose flag
        self.verbose = verbose
        
        # Edit tags
        self.tags = self.__init_tags()

        # Final data of container
        self.main_data = []

    def __init_tags(self)->list:
        """
        Method which initialize tag list.
        Init tags are: "SeriesDescription", "SliceThickness

        """
        # Define tags
        _tag_list = [
            ["Modality", "0008|0060"],
            ["BodyPartExamined", "0018|0015"],
            ["SeriesDescription", "0008|103e"],
            ["SliceThickness", "0018|0050"],
            ["ImageOrientationPatient", "0020|0037"]
        ]
        return _tag_list

    def __obtain_series_dirs(self, 
                             path_dir: str = None)->list:
        """
        Method which seeks for the directories inside the main dir. It lists them
        and returns them. If there is no subdirectories then it is assumed that given
        directory is the one which contain series.

        Args:
            * path_dir, str, path to input dir in which we seek for series
        
        Return: list, list containing either given path (assuming that it contain series)
            or list of subdirectories in given dir assuming they represent series.
        """
        # Storage
        _directories_list = []

        # Find all directories inside the given directory which contain dcm file
        # inside
        for _root, _dirs, _files in os.walk(path_dir):
            if any(_file.endswith('.dcm') for _file in _files):
                _directories_list.append(_root)
        # Test
        if not _directories_list:
            if self.verbose:
                print(f"Found 1 potential series dir in {path_dir}")
            return [path_dir]
        else:
            if self.verbose:
                print(f"Found {len(_directories_list)} potential series dirs in {path_dir}")
            return _directories_list
        
    #TODO
    def __check_series_quality(self,
                        numpy_image:np.array = None,
                        tags_dict:dict = None)->bool:
        """
        Method which checks the validity of the extracted series. Current
        implemented policies:
                * Check for 3d and 3D in "Series decription tag 0008|103e"
                * Check for the shape of the extracted array, it must has 3 
                dimensions and every dimension must have more than 10 slices
        
        Args:
            * numpy_image, np.array, image extracted from the series
            * tags_dict, dict, dictionary obtained from the series based on tags
            defined for te containers

        Returns:
            * validity, boolean, True if the series is valid or False if
            the series should be discarded based on any policy.
        """
      
        # Check the series validity(3D in series tag) -> 3D not allowed TODO
        if tags_dict["0008|103e"][0] != None:
            if "3D" in tags_dict["0008|103e"][0] or "3d" in tags_dict["0008|103e"][0]:
                if self.verbose:
                    print(f"\tERROR: 3D is not supported yet(Tag)!!!")
                
                # Policy break
                return False

        # Check the series validity(volume shape) -> 3D not allowed TODO
        if len(numpy_image.shape) > 3:
            if self.verbose:
                print(f"\tERROR: 3D is not supported yet(Volume shape has more than 3 dimensions)!!!")
            return False
        
        # Check for number of slices
        _x, _y, _z = numpy_image.shape
        if _x < 10 or _y < 10 or _z < 10:
            if self.verbose:
                print(f"\tERROR: Volume contains lass than 10 slices in one of the dimensions ({_x, _y, _z})!!!")
            return False
        
        # If all policies passed
        return True
    
    def __fix_data_series(self, series_list: list = None)->list:
        """
        Method for filtering slices with wrong size. The idea is to do it by majority vote and then drop ones which are in minority.

        Args:
            * series_list, list, list of series-->dir paths

        Return:
            * filtered_list, list, filtered series 
        """
        # Storage list
        _filtered_list = []
        _size_list = []
        # Iterate over series
        for _dcm in series_list:
            # Obtain information
            _reader = sitk.ImageFileReader()
            _reader.SetFileName(_dcm)
            _reader.ReadImageInformation()
            
            # Obtain size
            _size_list.append(_reader.GetSize())

        # Filter sizes
        _size_counter = Counter(_size_list)
        _most_common_size = _size_counter.most_common(1)[0][0]
        _binary_array = [1 if _dim == _most_common_size else 0 for _dim in _size_list]

        # Appliy filter
        _filtered_list = [_item for _item, _flag in zip(series_list, _binary_array) if _flag == 1]

        # Return
        return _filtered_list



    def edit_tags(self, 
                  tags_list: list = [], 
                  mode: chr = 'a' ):
        """
        Method to add new tags to the list
        
        Args:
            * tags_list, list, list of tags to add to current list of tags or a
            list which will swap the current tag list. Tag is in format [str1, str2] where
            str1 is description and str2 is tag number xxxx|xxxx
            * mode, chr, 'a' for appending to existing tag list, or 'o' to owerride
            the current tag list and swipe it with a new one 

        """
        # Check mode
        assert mode in ['a', 'o'], f"Invalid mode operator, only 'a' and 'o' are supported but get {mode}!"

        # Check tag modality
        if mode == 'a':
            self.tags += tags_list
        if mode == 'o':
            self.tags = tags_list

    def obtain_data_from_series(self, path_dir: str = None, mode = 'a')->list:
        """
        Method which seeks for the series inside every dir. Then it extract necessary information and 
        returns formated directory for each worked series.

        Args:
            * path_dir, str, path to input dir in which we seek for series
            * mode, chr, 'a' for appending to existing tag list, or 'o' to owerride
            the current tag list and swipe it with a new one 

        Returns: list, list of parsed series where each series is a dictionary containing all necessary
        information of the series
        """
        # Check if path_dir exists and if path_dir is a dir
        assert path_dir != None and os.path.exists(path_dir), f"Directory {path_dir} does not exists!"

        # Check mode validity
        assert mode in ['a', 'o'], f"Invalid mode operator, only 'a' and 'o' are supported but get {mode}!"

         # Obtain potential series candidates
        _series_dirs = self.__obtain_series_dirs(path_dir)

        # Define storage
        _main_storage = []

        # Define reader
        _series_reader = sitk.ImageSeriesReader()

        # Go through dirs
        if self.verbose:
            print("--------------------------------------------------------------------------------------")
        for _dir_name in _series_dirs:
            # Obtain all series IDs inside _dir
            _series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(_dir_name)
            if self.verbose:
                print(f"In dir: {_dir_name} found {len(_series_IDs)} series.")
            
            # Go through each series to obtain data
            for _i, _series_ID in enumerate(_series_IDs):
                # Info
                if self.verbose:
                    print(f"Working on series {_i+1}/{len(_series_IDs)}")
                
                # Obtaining sitk image container
                _series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(_dir_name, _series_ID)
                _series_reader.SetFileNames(_series_file_names)
                _series_reader.MetaDataDictionaryArrayUpdateOn()
                _series_reader.LoadPrivateTagsOn()
                try: 
                    _image = _series_reader.Execute()
                except:
                    print(f"\tWARNING: Exception thrown by series_reader! Fixing it!")
                    _series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(_dir_name, _series_ID)
                    _series_file_names = self.__fix_data_series(_series_file_names)
                    _series_reader.SetFileNames(_series_file_names)
                    _series_reader.MetaDataDictionaryArrayUpdateOn()
                    _series_reader.LoadPrivateTagsOn()
                    try:
                        _image = _series_reader.Execute()
                    except:
                        print("\t ERROR: Exception thrown by series_reader! No help!")
                        continue

                # Obtain tags
                _tags_dict = {}
                for _tag in self.tags:
                    # Grab data from series
                    try:
                        _value = _series_reader.GetMetaData(0, _tag[1])
                    except:
                        _value = None
                    _tags_dict[_tag[1]] = [_value, _tag[0]]

                # Extracting information from the image
                _image_info_dict = {'size': _image.GetSize(),
                                    'spacing': _image.GetSpacing(),
                                    'origin': _image.GetOrigin(),
                                    'direction': _image.GetDirection()
                }
                             
                # Transfering image 
                _image_numpy_array = sitk.GetArrayFromImage(_image)                
                
                if self.__check_series_quality(numpy_image = _image_numpy_array,
                                               tags_dict = _tags_dict) == False:
                    continue
                
                # Reshape numpy array
                _image_numpy_array = rearrange(_image_numpy_array, 'z y x -> x y z')

                # Export info
                _series_export_dict = {
                    'main_dir': _dir_name,
                    'series_ID': _series_ID,
                    'series_tags': _tags_dict,
                    'series_image_info': _image_info_dict,
                    'image': _image_numpy_array
                }

                # Add to export list
                _main_storage.append(_series_export_dict)
        
        # Save/Return Info
        
        if self.verbose:
            print(f"Total number of found series is: {len(_main_storage)}")
            print("--------------------------------------------------------------------------------------")
        
        # Store the data
        if mode == 'a':
            self.main_data += _main_storage
        if mode == 'o':
            self.main_data = _main_storage
        return _main_storage
        
    def load_data_from_directory(self, load_dir_path:str = None, mode = 'a')->list:
        """
        Function which lists directory and tries to load all data into proper dictionary and 
        build container from exported container data.

        Args:
            * load_dir_path, str, name of the dictionary which were the results are stored
        Returns: list,  list containing either given path (assuming that it contain series)
            or list of subdirectories in given dir assuming they represent series.
            * mode, chr, 'a' for appending to existing tag list, or 'o' to owerride
            the current tag list and swipe it with a new one
        """
        # Check if path_dir exists and if path_dir is a dir
        assert load_dir_path != None and os.path.exists(load_dir_path), f"Directory {load_dir_path} does not exists!" 

        # Check mode validity
        assert mode in ['a', 'o'], f"Invalid mode operator, only 'a' and 'o' are supported but get {mode}!"

        # obtain all files in directory
        _files_in_dir = os.listdir(load_dir_path)

        # remove suffixes and remove duplicates
        _files_to_load = set([os.path.splitext(_file)[0] for _file in _files_in_dir])
        
        # Storage
        _main_storage = []

        # Build up storage
        for _file in _files_to_load:
            # Load json
            _full_file_path = os.path.join(load_dir_path, _file)
            with open(f"{_full_file_path}.json", 'r') as _json_file:
                _json_data_dict = json.load(_json_file)
            
            # Load image
            _image = np.load(f"{_full_file_path}.npy")

            # Merge
            _json_data_dict["image"] = _image

            # Store
            _main_storage.append(_json_data_dict)
        
        # Store the data
        if mode == 'a':
            self.main_data += _main_storage
        if mode == 'o':
            self.main_data = _main_storage
        return self.main_data
    
    def export_data_to_directory(self, 
                                 export_dir_path:str = None, 
                                 export_id: str = None,
                                 export_nifti: bool = True, 
                                 export_npy: bool = True):
        """
        Method for exporting data to a provided directory. The export format for one series is
        'dir_series.json' for data and 'dir_series.npy' for images. It override content with same name
        and create dir if it does not exists.
        
        Args:
            * export_dir_path, str, path to dir where files will be stored. 
            * export_id, str, if provided this is the id of the case (ovedrides everything)
            * export_nifti, bool, if true it will export nifti of the container
            * export_npy, bool, if true it will export numpy 
        """
        # Check if directory exists -> create if not
        if not os.path.exists(export_dir_path):
            os.makedirs(export_dir_path)
            if self.verbose:
                print(f"Directory {format(export_dir_path)} created.")
        else:
            if self.verbose:
                print(f"Directory {format(export_dir_path)} already exists, appending to it!")

        # Go thorough series and export them
        if self.verbose:
            print("--------------------------------------------------------------------------------------")
        for _series in self.main_data:
            # Create export name
            try:
                _main_name = [_s for _s in _series["main_dir"].split(os.sep) if "case" in _s][0]
            except:
                _main_name = ""

            if _main_name == "":
                _main_name = "case-default" 

            if export_id != None:
                _main_name = export_id

            _base_name ="__".join([_main_name, _series["series_ID"]])
            if self.verbose:
                print(f"Exporting {_base_name}")
            
            # Create json series without image
            _series_for_json = _series.copy()
            del _series_for_json["image"]

            # Write JSON to a file
            _save_name = os.path.join(export_dir_path, f"{_base_name}.json")
            with open(_save_name, 'w') as _json_file:
                json.dump(_series_for_json, _json_file, indent=4)

            # Write image as npy
            if export_npy:
                _save_file = os.path.join(export_dir_path, f"{_base_name}.npy")
                np.save(_save_file, _series["image"])   

            # Export nifti if needed
            if export_nifti:
                # Reconstruct image
                _image = reconstruct_image(input_series_dict=_series, display=False)
                _save_file = os.path.join(export_dir_path, f"{_base_name}.nii.gz")
                sitk.WriteImage(_image, _save_file)
    
    def export_images_to_directory(self, export_dir_path:str = None,
                                   export_id: str = None,
                                   resample_spacing: list = [1,1,1],
                                   lower_bound: int = -1024,
                                   upper_bound: int = 1500,
                                   histogram_reduction_params: dict = {"search_region": 400,
                                                        "reduction_factor": 50,
                                                        "number_of_bins": 500,
                                                        "upper_bound": 1500}):
        """
        Method for exporting images from the series. It exports three images:
        orignal image, image transfered to coronal, and histogram reduced image.

        Args:
            * export_dir_path, str, path to the dir where images will be exported.
            * export_id, str, if provided this is the id of the case (ovedrides everything)
            * resample_spacing, list, see function resample_image from Utils.Containers for more info.
            The default parameters are 1x1x1.
            * lower_bound, int, inital corropping value for the volume. Everything < lower_bound = lower_bound
            * upper_bound, int, initial cropping value for the volume. Everything > upper_bound = upper_bound
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
              for more info. The selected params were the best one.
        """
         # Check if directory exists -> create if not
        assert export_dir_path != None, f"Export dir must not be empty!!!"
        if not os.path.exists(export_dir_path):
            os.makedirs(export_dir_path)
            if self.verbose:
                print(f"Directory {format(export_dir_path)} created.")
        else:
            if self.verbose:
                print(f"Directory {format(export_dir_path)} already exists, appending to it!")
        
        # Go thorough series and export them
        if self.verbose:
            print("--------------------------------------------------------------------------------------")
        
        for _series in self.main_data:
            # Create export name
            try:
                _main_name = [_s for _s in _series["main_dir"].split(os.sep) if "case" in _s][0] 
            except:
                _main_name = ""
            if _main_name == "":
                _main_name = "case-default"
            if export_id != None:
                _main_name = export_id
            _base_name ="__".join([_main_name, _series["series_ID"]])
            if self.verbose:
                print(f"Exporting {_base_name}")

            # Transfer to coronal
            # Reconstruct image
            _reconstructed_image = reconstruct_image(_series)
            
            # Resample image
            _resampled_image = resample_image(image = _reconstructed_image,
                                              new_spacing = resample_spacing)
            # Go to numpy
            _resampled_item = transfer_to_numpy(input_image = _resampled_image)
            
            # Bound the volume
            _resampled_item['image'] = bound_volume(volume = _resampled_item['image'],
                                                    lower_bound = lower_bound,
                                                    upper_bound = upper_bound)
                        
            # Export origin image
            _summed_2d_array = np.sum(_resampled_item['image'], axis=2)
            
            # Apply colormap gray from matplotlib
            _norm = plt.Normalize(vmin = _summed_2d_array.min(), vmax = _summed_2d_array.max())
            _image_norm = plt.cm.gray(_norm(_summed_2d_array))
            _image_norm = (_image_norm[:,:, 0] * 255).astype(np.uint8)
            _image_norm = Image.fromarray(_image_norm.T)
            _image_norm.save(os.path.join(export_dir_path, f"{_base_name}_original_image.png"))
                        
            # Export coronal image
            _image = transfer_to_coronal(image = _resampled_item['image'], 
                        tags = _series['series_tags'])
            _summed_2d_array = np.sum(_image, axis=2)
            # Apply colormap gray from matplotlib
            _norm = plt.Normalize(vmin = _summed_2d_array.min(), vmax = _summed_2d_array.max())
            _image_norm = plt.cm.gray(_norm(_summed_2d_array))
            _image_norm = (_image_norm[:,:, 0] * 255).astype(np.uint8)
            _image_norm = Image.fromarray(_image_norm.T)
            _image_norm.save(os.path.join(export_dir_path, f"{_base_name}_coronal_image.png"))

            # Do histogram reduction
            _reducted_image, _, _ = histogram_reduction(image = _image, 
                                                        search_region = histogram_reduction_params['search_region'],
                                                        reduction_factor = histogram_reduction_params['reduction_factor'],
                                                        number_of_bins = histogram_reduction_params['number_of_bins'],
                                                        upper_bound = histogram_reduction_params['upper_bound'])
            # Export reducted image
            _summed_2d_array = np.sum(_reducted_image, axis=2)
            # Apply colormap gray from matplotlib
            _norm = plt.Normalize(vmin = _summed_2d_array.min(), vmax = _summed_2d_array.max())
            _image_norm = plt.cm.gray(_norm(_summed_2d_array))
            _image_norm = (_image_norm[:,:, 0] * 255).astype(np.uint8)
            _image_norm = Image.fromarray(_image_norm.T)
            _image_norm.save(os.path.join(export_dir_path, f"{_base_name}_reducted_image.png"))

    @property
    def data(self)->list:
        """
        Conviniece method which returns main data of container. 
        
        Return: list, list containing per slice obtained data by method
        obtain_data_from_series
        """
        return self.main_data
        
    @property
    def tag_list(self):
        """
        Method which returns tags which will be extracted for each series
        """
        return self.tags

    def __getitem__(self, ndx)->dict:
        """
        For iterating through data

        Returns: dict, one series stored in self.main_data (container)
        """
        assert self.main_data, f"Data must not be empty, load data into container!"
        return self.main_data[ndx]

    def __len__(self):
        """
        Utility function which return amount of data stored in container
        """
        return len(self.main_data)
    
    def __str__(self):
        """
        Info on how to use container
        """
        _str = "LOAD DATA: To use containers load data with methods: obtain_data_from_series or load_data_from_directory. \n"
        _str += "TAGS: To select tags use method edit tags. To find which tags are included use property tags_list. \n"
        _str += "EXPORT: To export data to directory use method export_data_to_directory. \n"
        _str += "IMAGE: To obtain image use utilty function reconstruct_image and provide one series from container to it. \n"

        return _str