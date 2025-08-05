import os
import pandas as pd
import cv2
import json
# Image dir
_input_dir =  r"C:\Users\CH258598\Desktop\Verify"

# Create containers

# Load images
_images = []
_jsons = []
for _root, _, _files in os.walk(_input_dir):
    for _file in _files:
        if _file.endswith("_reducted_image.png"):
            _images.append(os.path.join(_root, _file))
            _json = _file.replace(".png", "_prediction.json")
            _jsons.append(os.path.join(_root, _json))
# Load images
_images_dict = {}
for _i, _img_path in enumerate(_images):
    _img = cv2.imread(_img_path)
    _images_dict[_i] = _img

# Label
_good_list = []
_wrong_prediction = []
_wrong_list = []

for _i, _json in enumerate(_jsons):
    os.system('cls')
    print(f"Image_Name: {_images[_i]}")
    print("------------------------------------------------------------------------------")
    with open(_json, "r") as file:
        _data = json.load(file)
    for _key in _data:
        print(f"{_key}:{_data[_key]}\n")

    print("------------------------------------------------------------------------------")
    _img = _images[_i]
    cv2.namedWindow("Verify", cv2.WINDOW_NORMAL) 

    # Resize the window
    cv2.resizeWindow("Verify", 750, 750)

    # Move the window to the specified position
    cv2.moveWindow("Verify", 400, 75)

    cv2.imshow("Verify", _images_dict[_i])
    _key_pressed = cv2.waitKey(0)
    
    if _key_pressed == ord('q'):
        _good_list.append(_img)
        cv2.destroyAllWindows()

    if _key_pressed == ord('w'):
        _wrong_prediction.append(_img)
        cv2.destroyAllWindows()

    if _key_pressed == ord('e'):
        _wrong_list.append(_img)
        cv2.destroyAllWindows()

    if _key_pressed == ord('r'):
        cv2.destroyAllWindows()
        break
        

print("------------------------------------------------------------------------------")
print("GoodList:", _good_list)
print("------------------------------------------------------------------------------")
print("WrongPredictionList:", _wrong_prediction)
print("------------------------------------------------------------------------------")
print("WrongList:", _wrong_list)
print("------------------------------------------------------------------------------")

_dict = {"GoodList": _good_list,
         "WrongPredictionList": _wrong_prediction,
         "WrongExportList": _wrong_list}

with open("lists.json", "w") as json_file:
    json.dump(_dict, json_file, indent=4)



