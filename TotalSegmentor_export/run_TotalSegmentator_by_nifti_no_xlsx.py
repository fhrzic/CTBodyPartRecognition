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


filename_xlsx = 'TotalSegmentatorInfo.xlsx'
df = pd.read_excel(open(filename_xlsx, 'rb'), sheet_name='Task')

dff = df.copy()
dff['LabelMerged'] = dff['Label']      

tasks = ["total", "appendicular_bones"]


filename_xlsx_out = os.path.join(os.getcwd(), 'LabelVolume_TotalSegmentator.xlsx')

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
    


    ### get ID based on filename

    ### run TotalSegmentatior for each of specified tasks
    info = {}
    for task in tasks:
        _file_path = filename.replace('.nii.gz', '_TotalSegmentatior_'+task+'.nii')
        if os.path.isfile(_file_path):
            print(f"Skipping {itr} {task}, file already exported!")
            continue
        info[task] = {}
        info[task]['status'], info[task]['filename'], info[task]['AnalysisTime'] = TotalSegmentator(filename, task=task, get_time=True)
