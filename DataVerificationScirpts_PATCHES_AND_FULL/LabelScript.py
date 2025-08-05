import os
import pandas as pd
import cv2

# Image dir
_src_image_dir =  r"Images2Label"

# Pandas
_src_xlsx = r"SourceDataset.xlsx"
_df = pd.read_excel(_src_xlsx, sheet_name = "mapping")

# Create containers
_good_list = []
_noughty_list = []

# Load images to the memory [Batches of 500]
_labeling_batch = 20
_begin_index = (_labeling_batch - 1) * 500 
_end_index = (_labeling_batch) * 500 - 1

# Load images
_images = {}
for _img_name in _df["Remaped_paths"][_begin_index:_end_index]:
    _img = cv2.imread(os.path.join(_src_image_dir, _img_name))
    _images[_img_name] = _img

# Label
for _key in _images:
    cv2.namedWindow(_key, cv2.WINDOW_NORMAL) 

    # Resize the window
    cv2.resizeWindow(_key, 750, 750)

    # Move the window to the specified position
    cv2.moveWindow(_key, 100, 100)

    cv2.imshow(_key, _images[_key])
    _key_pressed = cv2.waitKey(0)
    
    if _key_pressed == ord('y'):
        _good_list.append(_key)
        cv2.destroyAllWindows()

    if _key_pressed == ord('n'):
        _noughty_list.append(_key)
        cv2.destroyAllWindows()

    if _key_pressed == ord('e'):
        break
        cv2.destroyAllWindows()

# Build df and export
_df_1 = pd.DataFrame()
_df_1["good"] = _good_list
_df_2 = pd.DataFrame() 
_df_2["noughty"] = _noughty_list
_df = pd.concat([_df_1, _df_2], axis=1) 
_df.to_excel(f"batch_{_labeling_batch}.xlsx", index = False, sheet_name = "data")







