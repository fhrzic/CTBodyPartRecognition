import os
import time
import datetime
import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy
import glob

def ArrayToImage(arr, imgo):
    '''
    Convert array to image and setup the properties based on the original image using SimpleITK
    
    Parameters
    ----------
    arr:       np.ndarray(m, n, p) <float> 
               Array containing the image values
    imgo:      <SimpleITK.SimpleITK.Image>
               The original image
    
    Returns
    -------
    img:       <SimpleITK.SimpleITK.Image>
               The reconstructed image from array based on the properties of the original image
    
    '''
    
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(imgo.GetSpacing())
    img.SetOrigin(imgo.GetOrigin())
    img.SetDirection(imgo.GetDirection())
    return img

def CalculateSegmentVolume(image_array, seglabel, volume_per_voxel):
    '''
    Calculate the volume of the segmentation in [mm^3]
    
    Parameters
    ----------
    image_array:          np.ndarray(m, n, p) <uint8> 
                          Array containing the image segmentation labels
    seglabel:             <int>
                          The value of the segmentation that its volume is calculated
    volume_per_voxel:     <float>
                          The volume of each voxel in [mm^3]
    
    Returns
    -------
    volume:               <float>
                          The volume of the segmentation in [mm^3]
    
    '''
    volume = (image_array == seglabel).sum() * volume_per_voxel
    return volume

def RemoveValueFromArray(array, value):
    '''
    Remove a value from an array
    
    Parameters
    ----------
    array:            np.ndarray 
                      Input array with an arbitrary shape
    value:            <float; int; etc>
                      The value which will be removed from the array
    
    Returns
    -------
    array_modified:   np.ndarray(n,1) 
                      Output array which is flatten of input array excluding the value
    
    '''
    
    array_modified = np.delete(array, np.where(array == value))
    return array_modified


# Storage
filename_xlsx_out = 'Output-Labels.xlsx'

# Get tasks
filename_xlsx = 'TotalSegmentatorInfo.xlsx'
df = pd.read_excel(open(filename_xlsx, 'rb'), sheet_name='Task')
dff = df.copy()
dff['LabelMerged'] = dff['Label']      
tasks = ["total", "appendicular_bones"]


# Get input nii files
input_dir_path = r"/mnt/HDD/NMDD/Exported_To_X-ray/"
_nifti_file_list = []
for _root, _dirs, _files in os.walk(input_dir_path):
    for _file in _files:
        if _file.endswith('.nii'):
            _nifti_file_list.append(os.path.join(_root, _file))

filenames = _nifti_file_list
#filenames = glob.glob(os.path.join(os.getcwd(), 'CT_unique')+'/*.nii.gz')
filenames.sort()

# Generate sublists
sublists = [filenames[i:i+2] for i in range(0, len(filenames), 2)]

# Discard uncompleted cases
for i, item in enumerate(sublists):
    if len(item) !=2 :
        del sublists[i]

### find amount of shift needed for the labels of each task for merging
LabelShift = {}
summ = 0
for task in df['Task'].unique():
    LabelShift[task] = summ
    dff.loc[dff['Task'] == task, 'LabelMerged'] = dff.loc[dff['Task'] == task, 'LabelMerged'] + LabelShift[task]
    summ = summ + len(df[df['Task'] == task])

# Calculate volumes
final_list = []
for itr, exported_niis in enumerate(sublists):
    print(f"Processing {itr+1}/{len(sublists)}")

    ### get the analysis time
    start_time = time.time()
    
    ### get ID based on filename
    ID = os.path.basename(exported_niis[0]).split("_Total")[0]

    ### run TotalSegmentatior for each of specified tasks
    
    ### get segmentation information from the image segmented by TotalSegmentatior
    status = 0
    info = {}
    for task in tasks:
        info[task] = {}
    for i, task in enumerate(tasks):

        info[task]['image'] = sitk.ReadImage(exported_niis[1-i])
        info[task]['array'] = sitk.GetArrayFromImage(info[task]['image'])
        info[task]['spacing'] = info[task]['image'].GetSpacing()
        info[task]['voxel_volume'] = float(np.prod(info[task]['spacing'], dtype=np.float64))
        
        ### find the volume of each label
        labels      = np.unique(info[task]['array']) # get labels in the image
        labels      = RemoveValueFromArray(labels, value=0) # remove background from labels
        labelstasks = df[df['Task'] == task]['Label'].tolist()
        info[task]['volume'] = {}
        for labelstask in labelstasks:
            if labelstask in labels:
                info[task]['volume'][labelstask+LabelShift[task]] = CalculateSegmentVolume(info[task]['array'], labelstask, info[task]['voxel_volume'])
            else:
                info[task]['volume'][labelstask+LabelShift[task]] = None

    ### merge dictionaries for volume and convert it to list
    mydict = {}
    mydict['ID'] = ID
    for task in tasks:
        mydict = mydict | info[task]['volume']
    
    mylist = list(mydict.values())
    final_list.append(mylist)

dfl = pd.DataFrame(final_list, columns=list(mydict.keys()))

writer = pd.ExcelWriter(filename_xlsx_out)
dfl.to_excel(writer, sheet_name='Volume', engine='xlsxwriter')
dff.to_excel(writer, sheet_name='Task', engine='xlsxwriter')
writer.close()
