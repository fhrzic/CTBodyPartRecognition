from .Containers import *
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from einops import rearrange
import os
import nibabel as nib


def plot_image_hist(image: np.array = None, bins:int = 250):
    """
    Plot histogram of image (np array)

    Args:
        * image, np.array, input image
        * bins, int, number of bins in histogram. Greater the number is the
        better resolution is obtained.
    """
    _flattened_array = image.flatten()
    _b, _bins, _patches = plt.hist(_flattened_array, bins=250, alpha=0.7, color='blue')
    plt.title('Histogram of 3D Array Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def export_nifti_to_axial(nifti_path: str = None, output_path: str = None):
    """
    Function which export nifti file to axial plane by using nibabel library.
    Namel, doing the same in the simple ITK is a nightmare. 

    Args:
        * nifti_path, str, input image from simple ITK
        * output_path, str, if left None the same file will be overriden
    """
    # Load nifit
    _loaded_nifti = nib.load(nifti_path)
    
    # Defne target orinetation
    _target_ornt = nib.orientations.axcodes2ornt("LPS") 
    
    # Read original orientation
    _original_ornt = nib.orientations.io_orientation(_loaded_nifti.affine)  
    
    # Caluclate transformation from original orientation to target orientation
    _transformation = nib.orientations.ornt_transform(_original_ornt, _target_ornt) 
    
    # Apply transformation
    _oriented_nifti = _loaded_nifti.as_reoriented(_transformation)
    
    #Save   
    if output_path != None:
        nib.save(_oriented_nifti, output_path)
    else:
        nib.save(_oriented_nifti, nifti_path)


def transfer_to_coronal(image: np.array = None, 
                       tags: dict = None)->np.array:
    """
    Function which generate coronal image projection from any other.
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
        image = rearrange(image, 'x y z -> z y x')
        return image

    # Axial plane TO BE TESTED 
    if np.array_equal(_tag_value, _axial_plane):
        image = rearrange(image, 'x y z -> x z y')
        image = image[:, ::-1, :]
        return image
    
    # Not found any matchin plane -> returning the same image
    return image

def histogram_reduction(image: np.array = None, 
                       number_of_bins: int = 250, 
                       upper_bound: int = 1200,
                       search_region: int = 400,
                       reduction_factor: int = 50):
    """
    Fucntion for conversion to 2D by utilizing sumation over z-axis
    It applies filtering in Hounsfield domain and then summ over z axis to obtain 
    2D representation. Filtering is done on histogram level by reducin the histogram for
    reduction_factor inside search_region discretize by number_of_bins. The last positive
    number after filtering is taken as lower bound, while upper_bound is given as an 
    argument.

    Args:
        * image, np.array, image obtained from sitk.GetArrayFromImage with correct channel order.
        * number_of_bins, int, number of bins in which the histogram will be devided.
        * upper_bound, int, all values above upper_bound one will be set to the upper_bound value
        * search_region, int, +-search_region is ROI in which the threshold will be found
        * 

    Retruns: 
        * image, np.array, scalled image based on histogram reduction
        * [_lower_bound, _upper_bound], integers, value of boundings arround region 
        of interest
        * min_value, int, minimal value found in the image
    """
    # Create histogram of values
    _flattened_array = image.flatten()
    _hist, _bin_edges = np.histogram(_flattened_array, 
                                     bins = number_of_bins)
    
    # Find cropping roi
    _low_roi_index = np.argmax(_bin_edges>-search_region)
    _high_roi_index = np.argmax(_bin_edges>search_region)

    # Make filtering more robust
    _hist[:_low_roi_index] = 0
    _hist[_high_roi_index:] = 0

    # Reduce and round to 0
    _hist = _hist - np.percentile(_hist[_hist != 0], reduction_factor)
    _hist[_hist < 0] = 0

    # Find the lower bound
    _lower_bound = _bin_edges[np.argwhere(_hist > 0)[-1]]
    # Scale image and applay tolerance
    _min_value = np.min(image)
    image = np.where(image < _lower_bound, _min_value, image)
    image = np.where(image > upper_bound, upper_bound, image)
    
    # Return
    return image, [_lower_bound[0], upper_bound], _min_value

def bound_volume(volume: np.array = None, 
                 lower_bound: int = -1024, 
                 upper_bound: int = 1500)->np.array:
    """
    The function which bounds volume to the defined boundaries setting all values under
    the lower_bound to lower_bound value, and all avlues higher to upper_bound to upper_bound 
    value. 

    Args:
        * volume, np.array, volume which needs to be bounded.
        * lower_bound, int, lower bound/limit of the volume.
        * upper_bound, int, upper bound/limit, of the volume.
    
    Return:
        * volume, np.array, bounded volume
    """
    # Limit lower bound
    volume = np.where(volume < lower_bound, lower_bound, volume)
    
    # Limit upper bound
    volume = np.where(volume > upper_bound, upper_bound, volume)

    # Return
    return volume

def export_nifti_to_dirs(input_dir_path:str = None, 
                          export_dir_path: str = None, 
                          verbose: bool = False,
                          projection:dict = {"0020|0037": ["1\\0\\0\\0\\1\\0", "ImageOrientationPatient"]},
                          resample_spacing: list = [1,1,1],
                          lower_bound: int = -1024,
                          upper_bound: int = 1500,
                          histogram_reduction_params: dict = {"search_region": 400,
                                                        "reduction_factor": 50,
                                                        "number_of_bins": 500,
                                                        "upper_bound": 1500}):
    """
    Function which export ".niffti" files to directories. By export, it is meant original, coronal , reduced image.

    Args:
        * input_dir_path, str, path to the dir where nifti files are stored.
        * export_dir_path, str, path to the dir where images will be exported.
        * verbose, bool, if True, messages about the progress will be written.
        * projection, dict, hard codded to axial plane, but this must be provided in form of dict as presented.
        * resample_spacing, list, see function resample_image from Utils.Containers for more info.
        The default parameters are 1x1x1.
        * lower_bound, int, inital corropping value for the volume. Everything < lower_bound = lower_bound
        * upper_bound, int, initial cropping value for the volume. Everything > upper_bound = upper_bound
        * histogram_reduction_params, dict, see function histogram_reduction from Utils.Conversion
            for more info. The selected params were the best one.
    """
    # Check for the dirs
    assert export_dir_path != None and input_dir_path != None, f"Export and input dirs must not be empty!!!"

    # Create export dir if not existing
    if not os.path.exists(export_dir_path):
        os.makedirs(export_dir_path)
        if verbose:
            print(f"Directory {format(export_dir_path)} created.")
    else:
        if verbose:
            print(f"Directory {format(export_dir_path)} already exists, appending to it!")

    # Find all directories inside the given directory which contain dcm file inside
    _nifti_file_list = []
    for _root, _dirs, _files in os.walk(input_dir_path):
        for _file in _files:
            if _file.endswith('.nii.gz'):
                _nifti_file_list.append(os.path.join(_root, _file))
    if verbose:
        print(f"I have found: {len(_nifti_file_list)} nifti files!")
    
    # Iterate through nifti files and export them
    for _i, _nifti_file in enumerate(_nifti_file_list):
        # Progress
        if verbose:
            print(f"Wokring on {_i+1}/{len(_nifti_file_list)}")
        # Extract base name
        _base_name = os.path.splitext(os.path.splitext(os.path.basename(_nifti_file))[0])[0]
        
        # Read input image
        _input_image = sitk.ReadImage(_nifti_file)

        # Resample image
        _resampled_image = resample_image(image = _input_image,
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
                    tags = projection)
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