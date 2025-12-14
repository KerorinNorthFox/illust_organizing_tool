import os
import pathlib

def get_base_dir(current_file):
    proj_dir_name = "image_compare"
    real_path = os.path.realpath(current_file)
    
    while True:
        real_path = os.path.dirname(real_path)
        if proj_dir_name in os.path.basename(real_path):
            break
        
    return real_path

def get_absolute_path_if_not(base_path, path):
    real_path = path
    if not pathlib.Path(path).is_absolute():
        real_path = os.path.join(base_path, path)
    return real_path
