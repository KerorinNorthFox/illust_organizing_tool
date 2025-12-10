# ディレクトリ内で重複した画像を
# tmpディレクトリに移動するスクリプト
import os
import copy
import shutil
from detect_same_image import compare

script_path = os.path.realpath(__file__)
script_base = os.path.dirname(script_path)
proj_dir = os.path.dirname(script_base)
DIR = os.path.join(os.path.join(proj_dir, "dataset"), "mari")
TMP_DIR = os.path.join(DIR, "tmp")
print(f"TMP DIR: {TMP_DIR}")

THRESHOULD = 0.95

if not os.path.exists(TMP_DIR):
    print("mkdir")
    os.mkdir(TMP_DIR)
    
files_a = sorted(
    [f for f in os.listdir(DIR) if f.lower().endswith(("png", "jpg", "jpeg"))]
)

files_b = copy.deepcopy(files_a)

for a in files_a:
    print(f">>file A: {a}")
    for b in files_b:
        print(f">>file B: {b}")
        if a == b:
            print("Same file. Skip it.")
            continue

        path_a = os.path.join(DIR, a)
        path_b = os.path.join(DIR, b)

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print("file is already moved. Skip it.")
            continue

        score = compare(path_a, path_b)
        print(f">>Score is {score}")

        if score >= THRESHOULD:
           print("Moved.")
           shutil.move(path_b, TMP_DIR) 
