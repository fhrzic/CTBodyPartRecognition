import os
import shutil

def main(input_dir: str = None,
         output_dir: str = None):
    
    # Obtain files
    _dirs = [os.path.join(_output_dir, _d) for _d in os.listdir(_output_dir) if os.path.isdir(os.path.join(_output_dir, _d))]
    print(len(_dirs))

    # Obtain dirs
    for _dir in _dirs:
        _case = os.path.basename(_dir)
        try: 
            shutil.copy(os.path.join(input_dir, _case, "results.xlsx"), os.path.join(_dir, "results.xlsx"))
        except:
            shutil.copy(os.path.join(input_dir, _case, "results_WRONG.xlsx"), os.path.join(_dir, "results.xlsx"))
if __name__ == "__main__":
    _input_dir = "/mnt/SSD/Franko/Desktop/Train_BCH_NMDD_SAROS/FinalTrainingData/Full_NMDD"
    _output_dir = "/mnt/SSD/Franko/Desktop/FIXExport/NMDID_PATCHES"
    main(input_dir = _input_dir,
         output_dir = _output_dir)