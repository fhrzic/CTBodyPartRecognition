import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from .Containers import *
from .Conversion import *
import cv2
from more_itertools import chunked 
import scipy.ndimage
from .GenerateNPY import *
def process_volume(
                volume: sitk.Image = None,
                series_info_dict_path: str = None,
                new_spacing: tuple = (0.5, 0.5, 0.5), 
                lower_bound: int = -1024,
                upper_bound: int = 1500,
                histogram_reduction_params: dict = {"search_region": 400,
                                                "reduction_factor": 50,
                                                "number_of_bins": 500,
                                                "upper_bound": 1500})->list:
        """
        Method which transfer volumes to the 2D images

        Args:
            * volume, sitk.img, volume to be transfered
            * series_info_dict, str, path to json contatining series info
            * new_spacing, tuple, tuple of integers representing spacing for the simpleitk image
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
                for more info. The selected params were the best one.
                        * lower_bound, int, inital corropping value for the volume. Everything < lower_bound = lower_bound
            * upper_bound, int, initial cropping value for the volume. Everything > upper_bound = upper_bound
            * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
            for more info. The selected params were the best one.
        
        Returns:
            * list containing coronal and reducted np.arrays of the given 3D volume

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
        # Load projecton if not, it is assumed that it is in axial plane
        try:
            with open(series_info_dict_path, 'r') as _file:
                _series_info_dict = json.load(_file) 
            _series_info =  _series_info_dict["series_tags"]
        except:
            _series_info = {"0020|0037": ["1\\0\\0\\0\\1\\0", "ImageOrientationPatient"]}
        _image = transfer_to_coronal(image = _resampled_item['image'], 
                                     tags = _series_info)
        
        _coronal_image = np.copy(_image)
        # Do histogram reduction- just bound volume
        _reducted_image = bound_volume(volume = _image,
                                       lower_bound = lower_bound,
                                       upper_bound = upper_bound)
        #_reducted_image, _histogram_reduction_params, _min = histogram_reduction(image = _image, 
        #                                    search_region = histogram_reduction_params['search_region'],
        #                                    reduction_factor = histogram_reduction_params['reduction_factor'],
        #                                    number_of_bins = histogram_reduction_params['number_of_bins'],
        #                                    upper_bound = histogram_reduction_params['upper_bound'])
        # Return the reducted image
        return _coronal_image, _reducted_image     

def crop_and_export_to_NPY(volume_coronal: np.array = None,
                           volume_reducted: np.array = None,
                           export_dir: str = None, 
                           crop_cordinates: list = None):
        """
        Method which crops volume and, resizes it and export it to NPY file

        Args:
            * volume_coronal, np.array, intact volume to export
            * volume_reducted, np.array, volume reducted 
            * export_dir, str, name under which the volume will be saved
            * crop_cordinates, list, list containing 2 sublistst obtained by the function obtain_crop_cordinates_2D
        """
        # Copy arrays so the main ones are left intact
        _coronal_image = np.copy(volume_coronal)
        _reducted_image = np.copy(volume_reducted)
        
        # Crop
        if crop_cordinates != None:
            _x, _y = crop_cordinates
            _coronal_image = _coronal_image[_x[0]: _x[1], _y[0]:_y[1], :]
            _reducted_image = _reducted_image[_x[0]: _x[1], _y[0]:_y[1], :]

        # Resize them
        _coronal_image = resize_to_given_dimension(input_volume = _coronal_image,
                                                    new_size = 224)
        _reducted_image = resize_to_given_dimension(input_volume = _reducted_image,
                                                    new_size = 224)


        # Save coronal to the dir
        _name = os.path.join(export_dir, "coronal_bounded.npy")
        np.save(_name, _coronal_image) 

        # Export processed volume
        _name = os.path.join(export_dir, "reduced_bounded.npy")
        np.save(_name, _reducted_image) 
            
        # Export reducted image if necessary # THIS IS FOR TESTING PURPOSES, BUT THIS SHOULD GENERATE THE FINAL OUTPUT IMAGE
        # It will not be the same as the one in the file for 2d training due to the wrong
        # vmin and vmax parameters (calculate them on the whole image)
        _summed_2d_array = np.sum(_reducted_image, axis=2)
        #Apply colormap gray from matplotlib
        _norm = plt.Normalize(vmin = _summed_2d_array.min(), 
                              vmax = _summed_2d_array.max())



        _image_norm = plt.cm.gray(_norm(_summed_2d_array))
        _image_norm = (_image_norm[:,:, 0] * 255).astype(np.uint8)
        _image_norm = Image.fromarray(_image_norm.T)
        _image_norm.save("main_bounded.png")
        
        
def resize_to_given_dimension(input_volume: np.array = None,
                   new_size: int = 224)->np.array:
    """
    Method which resamples image and preserves image ratio by zeropadding the image
    
    Args:
        * input_volume, np.array, 3D input image which suppose to be resampled
        * new_size, int, desired size of a resized image [new_size, new_size, new_size]

    Returns:
        * np.array, resized image
    """
    # Get the original dimensions
    _original_shape = np.array(input_volume.shape)

    # Obtain max dim
    _max_dim = np.max(_original_shape)

    # Scaling factors
    _scaling_factor = new_size / _max_dim

    # Calculate new shape
    _new_shape = (_original_shape * _scaling_factor).astype(int)

    # Resize by using linear interpolation -- order == 1
    _resized_volume = scipy.ndimage.zoom(input_volume, 
                                        zoom=(_new_shape[0] / _original_shape[0],
                                              _new_shape[1] / _original_shape[1],
                                              _new_shape[2] / _original_shape[2]),
                                        order=1)  # order=1 for linear interpolation
    # Return
    return _resized_volume

def obtain_crop_cordinates_2D(main_image: np.array = None, patch: np.array = None):
    """
    Function which obtains coordinates of a patch and returns them

    Args:
        * main_image, np.array, main image (resample image that was outputed as reducted image)
        * patch, np.array, patch (cropped output)

    Returns:
        * Returns 2 tupples: (min_x, min_x + h) (min_y, min_y+w)
    """
    # Obtain matching patch
    _res = cv2.matchTemplate(main_image, patch, cv2.TM_CCOEFF_NORMED)
    
    # Get width and height of the image
    _w, _h = patch.shape[::-1]
    
    # Obtain 
    _, _, _, _max_loc = cv2.minMaxLoc(_res)
    
    # return
    #crop_image = _main_image[_max_loc[1]: _max_loc[1] + h , _max_loc[0] :  _max_loc[0] + w]
    return [_max_loc[1], _max_loc[1] + _h], [_max_loc[0], _max_loc[0] + _w]

def divide_list(input_list:list, number_of_sublists: int)->list:
    """
    Divide list in equal sublists

    Args:
        * input_list, list, list which must be divede into chunks
        * number_of_sublists, int, number represetning number of chunks

    Returns:
        * sublists, list, list of sublist 
    """
    # Calculate the number of elements per sublist
    _elements_per_sublist = len(input_list) // number_of_sublists
    _remainder = len(input_list) % number_of_sublists

    # Initialize starting index and result list
    _start = 0
    _sublists = []

    # Iterate over each sublist
    for _i in range(number_of_sublists):
        # Calculate the sublist size considering the remainder
        _sublist_size = _elements_per_sublist + (1 if _i < _remainder else 0)
        
        # Append sublist to the result list
        _sublists.append(input_list[_start:_start+_sublist_size])
        
        # Update the starting index for the next sublist
        _start += _sublist_size

    return _sublists



