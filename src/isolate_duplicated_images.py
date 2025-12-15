# ディレクトリ内で重複した画像を
# tmpディレクトリに移動するスクリプト
import os
import copy
import shutil
from detect_same_image import compare

from utils.path_solver import get_base_dir, get_absolute_path_if_not
from utils.settings import load_isolate_duplicated_images_data, JSON_PATH

target = load_isolate_duplicated_images_data(JSON_PATH)
TARGET = get_absolute_path_if_not(get_base_dir(__file__), target)
TMP_DIR = os.path.join(TARGET, "tmp")
print(f"TMP DIR: {TMP_DIR}")

THRESHOULD = 0.95

if not os.path.exists(TMP_DIR):
    print("mkdir")
    os.mkdir(TMP_DIR)
    
files_a = sorted(
    [f for f in os.listdir(TARGET) if f.lower().endswith(("png", "jpg", "jpeg"))]
)

files_b = copy.deepcopy(files_a)

for a in files_a:
    print(f">>file A: {a}")
    for b in files_b:
        print(f">>file B: {b}")
        if a == b:
            print("Same file. Skip it.")
            continue

        path_a = os.path.join(TARGET, a)
        path_b = os.path.join(TARGET, b)

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print("file is already moved. Skip it.")
            continue

        score = compare(path_a, path_b)
        print(f">>Score is {score}")

        if score >= THRESHOULD:
            print("Moved.")
            shutil.move(path_b, TMP_DIR) 
