import os
import shutil

os.makedirs("Verify")
_input_dir = "/mnt/HDD/CT"

_list = []

for _root, _, _files in os.walk(_input_dir):
    for _file in _files:
        if _file.endswith("_reducted_image.png"):
            _list.append(os.path.join(_root, _file))
            _json = _file.replace(".png", "_prediction.json")
            _list.append(os.path.join(_root, _json))


for _item in _list:
    _name = os.path.basename(_item)
    shutil.copy(_item, os.path.join("Verify", _name))

#/mnt/HDD/CT/100579940