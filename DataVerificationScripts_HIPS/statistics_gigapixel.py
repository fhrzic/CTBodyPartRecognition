import os
import pandas as pd
import cv2
import json
import random
from PIL import Image

def create_gigapixel_image(folder_paths: list, output_path:str, name: str = "def"):
    """
    Function which creates gigapixelimage.

    Args:
        * folder_paths, list, list containing all series 
        * output_path, str, path to the output dir.
        * name, str, save name
    """
    _images = []
    _max_width = 0
    _max_height = 0
    _total_height = 0
    _name = f"{name}_gigapixelimage.png"

    # Load images and calculate dimensions
    _folder_images = []    
    for _i, _folder_path in enumerate(folder_paths):
        
        _image = Image.open(_folder_path)
        _max_width = max(_max_width, _image.width)
        _max_height = max(_max_height, _image.height)
        _folder_images.append(_image)
        if (_i + 1 ) % 10 == 0:
            _total_height += _max_height
            _folder_images.append(_max_height)
            _max_height = 0
            _images.append(_folder_images)
            _folder_images = []

    # Create gigapixel image
    _gigapixel_image = Image.new('RGB', (_max_width * len(_images[0]), _total_height))
    
    # Paste images onto gigapixel image
    # Paste images onto gigapixel image
    _y_offset = 0
    for _folder_images in _images:
        _x_offset = 0
        _max_height = _folder_images[-1]
        _folder_images = _folder_images[:-1]
        for _img in _folder_images:
            _gigapixel_image.paste(_img, (_x_offset, _y_offset))
            _x_offset += _max_width
        _y_offset += _max_height
    
    # Save gigapixel image
    _gigapixel_image.save(os.path.join(output_path, _name))
    


def main():
    """
    Main scripts for the gigapixel generation
    """
    _json = r"lists.json"
    with open(_json, "r") as file:
        _data = json.load(file)


    _proejciton_dict = {}
    for _key in _data:
        print(_key, len(_data[_key]))
        _export_data = _data[_key]
        #_q = min(len(_export_data), 70)
        _export_data = random.sample(_export_data, 70)
        
        create_gigapixel_image(_export_data, "", _key)

        #for _item in _data[_key]:
        #    _name = os.path.basename(_item)
        #    print(_name)
        #print("----------------------------------------------------")

if __name__ == "__main__":
    main()