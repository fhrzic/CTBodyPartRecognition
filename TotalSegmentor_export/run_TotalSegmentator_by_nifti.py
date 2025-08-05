import os
import time
import datetime
import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy
import glob

def TotalSegmentator(filename, dir_save=None, basename_out=None, task="total", get_time=False):
    '''
    Segment CT scan by TotalSegmentator.
    
    Parameters
    ----------
    filename:      <str>
                   Name of the input image file. The format should be .nii.gz
    dir_save:      <str>
                   Directory of the output. If not specified, the results will be saved in the same directory as the input image.
    basename_out:  <str>
                   The name of the output file. If not specified, the name will be same as the <input basename>_TotalSegmentatior_<task>.nii.gz
    task:          <str>
                   The name of the TotalSegmentatior task.
                   The options are "total" "body" "lung_vessels" "cerebral_bleed" "hip_implant" "coronary_arteries" "pleural_pericard_effusion" 
                   "test" "appendicular_bones" "tissue_types" "heartchambers_highres" "face" "vertebrae_body"
    get_time:      <bool>
                   If True, the total calculation time is reported 
    
    Returns
    -------
    status:        <int>
                   Status of the system command.
    filename_out:  <str>
                   Full name (including the path) of the output file 
    
    '''
    
    if dir_save is None and basename_out is None:
        dir_save     = os.path.dirname(filename)
        basename     = os.path.basename(filename)
        basename     = basename.replace('.nii.gz', '')
        basename_out = basename+'_TotalSegmentatior_'+task+'.nii'
    elif dir_save is None and basename_out is not None:
        dir_save     = os.path.dirname(filename)
    elif dir_save is not None and basename_out is None:
        basename = os.path.basename(filename)
        basename = basename.replace('.nii.gz', '')
        basename_out = basename+'_TotalSegmentatior_'+task+'.nii'    
    
    filename_out = os.path.join(dir_save, basename_out)
    
    AnalysisTime = None
    if get_time:
        start_time = time.time()
    
    command = ["TotalSegmentator","-d", "gpu", "-i", filename, "-o", filename_out, "-ot nifti --ml", "-ta", task]
    command = ' '.join(command)
    status = os.system(command)
    
    if get_time:
        analysis_time = time.time() - start_time
        AnalysisTime  = str(datetime.timedelta(seconds=analysis_time)).split(".")[0]
    
    return status, filename_out, AnalysisTime

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

def FilterComponentsByVolume(image_array, seglabel, volume_per_voxel, threshold_volume_size, separate_componets=False):
    '''
    Remove a value from an array
    
    Parameters
    ----------
    image_array:               np.ndarray(m, n, p) <uint8> 
                               Array containing the image segmentation labels
    seglabel:                  <int>
                               The value of the segmentation that its volume is calculated
    volume_per_voxel:          <float>
                               The volume of each voxel in [mm^3]
    threshold_volume_size:     <float>
                               The volume threshold in [mm^3] indicates the volumes smaller than the threshold are removed
    separate_componets:        <bool>
                               If True, each label will be separated into isolated parts, and the volume of each part will be checked
    
    Returns
    -------
    image_array:               np.ndarray(m, n, p) <uint8> 
                               Output array which is cleaned up based on the volume
    fragments_array:           np.ndarray(m, n, p) <uint8> 
                               Output array which is the parts removed from the main image_array
    object_size:               <float> <dict>
                               Volume size of the seglabel. 
                               If separate_componets is False, object_size is <float>
                               If separate_componets is True, object_size is <dict>
                               
    '''
    
    fragments_array = image_array * 0
    if separate_componets:
        mask = image_array == seglabel
        lmap, num_objects = scipy.ndimage.label(mask.astype(int))
        object_size = {}
        if num_objects > 0:
            for object_id in range(1, num_objects + 1):
                #object_size[object_id] = (lmap == object_id).sum() * volume_per_voxel
                object_size[object_id] = CalculateSegmentVolume(lmap, object_id, volume_per_voxel)
                if object_size[object_id] < threshold_volume_size:
                    image_array[(lmap == object_id) & mask]     = 0
                    fragments_array[(lmap == object_id) & mask] = seglabel
        
        print('debug-- ',  'label: ', seglabel)
        print('        ', object_size)
        
    else:
        
        print('debug-- ',  'label: ', seglabel)
        object_size = CalculateSegmentVolume(image_array, seglabel, volume_per_voxel)
        if object_size < threshold_volume_size:
               print('      object_size:',  object_size)
               mask = image_array == seglabel
               image_array[mask]     = 0
               fragments_array[mask] = seglabel
    
    return image_array, fragments_array, object_size



filename_xlsx = 'TotalSegmentatorInfo.xlsx'
df = pd.read_excel(open(filename_xlsx, 'rb'), sheet_name='Task')

dff = df.copy()
dff['LabelMerged'] = dff['Label']      

tasks = ["total", "appendicular_bones"]


filename_xlsx_out = os.path.join(os.getcwd(), 'LabelVolume_TotalSegmentator.xlsx')

itr_backup = 20
filename_xlsx_out_backkup = os.path.join(os.path.dirname(filename_xlsx_out), os.path.basename(filename_xlsx_out).replace('.xlsx', '_BACKUP.xlsx'))

input_dir_path = r"/mnt/HDD/NMDD/Exported_To_X-ray/"

_nifti_file_list = []
for _root, _dirs, _files in os.walk(input_dir_path):
    for _file in _files:
        if _file.endswith('.nii.gz'):
            _nifti_file_list.append(os.path.join(_root, _file))

filenames = _nifti_file_list
#filenames = glob.glob(os.path.join(os.getcwd(), 'CT_unique')+'/*.nii.gz')
filenames.sort()

### logfile
logfile = os.getcwd() + '/' + 'out.log'
if os.path.exists(logfile):
  os.remove(logfile)

f_logfile = open(logfile, 'w')


### find amount of shift needed for the labels of each task for merging
LabelShift = {}
summ = 0
for task in df['Task'].unique():
    LabelShift[task] = summ
    dff.loc[dff['Task'] == task, 'LabelMerged'] = dff.loc[dff['Task'] == task, 'LabelMerged'] + LabelShift[task]
    summ = summ + len(df[df['Task'] == task])

final_list = []
for itr, filename in enumerate(filenames):
    print(f"Processing {itr+1}/{len(filenames)}")

    ### get the analysis time
    start_time = time.time()
    
    ### get ID based on filename
    ID = os.path.basename(filename).replace('.nii.gz','')
    
    ### run TotalSegmentatior for each of specified tasks
    info = {}
    for task in tasks:
        info[task] = {}
        info[task]['status'], info[task]['filename'], info[task]['AnalysisTime'] = TotalSegmentator(filename, task=task, get_time=True)
    
    ### get segmentation information from the image segmented by TotalSegmentatior
    status = 0
    for task in tasks:
        if info[task]['status'] == 0: # TotalSegmentator is completed successfully for the task
            info[task]['image'] = sitk.ReadImage(info[task]['filename'])
            info[task]['array'] = sitk.GetArrayFromImage(info[task]['image'])
            
            info[task]['spacing']      = info[task]['image'].GetSpacing()
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
            
        else:
            status = status - 1
            #info[task]['array_merge'] = 0
            info[task]['volume'] = {}
            labelstasks = df[df['Task'] == task]['Label'].tolist()
            for labelstask in labelstasks:
                info[task]['volume'][labelstask+LabelShift[task]] = None
    
    ### merge dictionaries for volume and convert it to list
    mydict = {}
    mydict['ID'] = ID
    for task in tasks:
        mydict = mydict | info[task]['volume']
    
    mylist = list(mydict.values())
    final_list.append(mylist)
    
    ### write to logfile
    analysis_time = time.time() - start_time
    AnalysisTime = str(datetime.timedelta(seconds=analysis_time)).split(".")[0]
    print("itr: %8s," %(str(itr)), " ID: %60s," %(str(ID)), " status: %4s," %(str(status)), '  AnalysisTime: ', "%8s " %(str(AnalysisTime)), file=f_logfile, flush=True )
    
    if itr % itr_backup:
        dfl = pd.DataFrame(final_list, columns=list(mydict.keys()))
        
        writer = pd.ExcelWriter(filename_xlsx_out_backkup)
        dfl.to_excel(writer, sheet_name='Volume', engine='xlsxwriter')
        dff.to_excel(writer, sheet_name='Task', engine='xlsxwriter')
        writer.close()
        

dfl = pd.DataFrame(final_list, columns=list(mydict.keys()))

writer = pd.ExcelWriter(filename_xlsx_out)
dfl.to_excel(writer, sheet_name='Volume', engine='xlsxwriter')
dff.to_excel(writer, sheet_name='Task', engine='xlsxwriter')
writer.close()

f_logfile.close()
