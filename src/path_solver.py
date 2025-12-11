import os
import pathlib

def get_base_dir(current_file):
    return os.path.dirname(os.path.dirname(os.path.realpath(current_file)))

def get_absolute_path_if_not(base_path, path):
    real_path = path
    if not pathlib.Path(path).is_absolute():
        real_path = os.path.join(base_path, path)
    return real_path
