from Utils.PatchExport import *
import argparse
import shutil
import os

def main(input_dir: str = None, 
         xlsx_dir: str = None,
         export_dir: str = None):
    """
    Main script which passes thorugh the all data in input dir and store it in export dir with added images, filters, corrected data etc.
    From PatchExport.

    Args:
        * input_dir, str, path to input dir which hold cases
        * export_dir, str, destination directory where everything will be stored
    """
    # List dir
    _input_dirs_list_names = os.listdir(input_dir)
    _input_dirs_list_full_names = [os.path.join(input_dir, _dir) for _dir in _input_dirs_list_names]

    # Transfer dirs and run inference
    _exported_dirs = []
    for _i, _input_dir in enumerate(_input_dirs_list_full_names[0:5]):

        # Copy all files
        _dst_name = os.path.join(export_dir, _input_dirs_list_names[_i])
        _exported_dirs.append(_dst_name)
        shutil.copytree(_input_dir, _dst_name, dirs_exist_ok=True)

        # Creat export container and work on the exports
        _my_container = patch_container(_dst_name)
        _my_container.refine_segmentation_volumes(_dst_name)
        _my_container.merge_blobs( export_path = _dst_name,
                         merging_criteria = "reduced_cluster_remaped")
        
        _xlsx_path = os.path.join(xlsx_dir, _input_dirs_list_names[_i], "results.xlsx")

        _my_container.filter_blobs(xlsx_path = _xlsx_path,
                          export_path =_dst_name)
        _my_container.generate_nifti(export_path = _dst_name)

if __name__ == "__main__":
    # Create the parser
    _parser = argparse.ArgumentParser()
    
    # Get arguments
    _parser.add_argument('input_dir', type = str, help = 'The input dir path')
    _parser.add_argument('xlsx_dir', type = str, help = "The input dir to xlsx path")
    _parser.add_argument('export_dir', type = str, help = 'The export dir path')

    _args = _parser.parse_args()

    # Call main
    main(input_dir = _args.input_dir,
         xlsx_dir = _args.xlsx_dir,
         export_dir = _args.export_dir)