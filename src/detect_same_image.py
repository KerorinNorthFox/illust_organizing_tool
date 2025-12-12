import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from path_solver import get_base_dir, get_absolute_path_if_not
from settings import load_detect_same_image_data, JSON_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
    ),
])

def _img_to_vec(path):
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model(tensor).squeeze().cpu().numpy()

    return vec

def _cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
def compare(path_a, path_b):
    v1 = _img_to_vec(path_a)
    v2 = _img_to_vec(path_b)
    score = _cosine_similarity(v1, v2)
    return score

def compare_all(dir_a, dir_b):
    files_a = sorted(
        [f for f in os.listdir(dir_a) if f.lower().endswith(("png", "jpg", "jpeg"))]
    )
    files_b = sorted(
        [f for f in os.listdir(dir_b) if f.lower().endswith(("png", "jpg", "jpeg"))]
    )

    results = []
    for i, file_a in enumerate(files_a):
        print(f"A > {file_a}")
        for j, file_b in enumerate(files_b):
            print(f"B > {file_b}")
            path_a = os.path.join(dir_a, file_a)
            path_b = os.path.join(dir_b, file_b)

            score = compare(path_a, path_b)
            results.append({
                "i": i,
                "j": j,
                "score": score,
                "file_name_a": file_a,
                "file_name_b": file_b,
                "file_path_a": path_a,
                "file_path_b": path_b,
            })
    return results

def display(result, save_dir=None):
    i = result["i"]
    j = result["j"]
    score = result["score"]
    file_a = result["file_name_a"]
    file_b = result["file_name_b"]
    path_a = result["file_path_a"]
    path_b = result["file_path_b"]

    img_a = Image.open(path_a)
    img_b = Image.open(path_b)
    
    plt.figure(figsize=(10,4))
    plt.suptitle(f"Similarity: {score:.8f}", fontsize=16)

    plt.subplot(1,2,1)
    plt.imshow(img_a)
    plt.title(file_a)
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(img_b)
    plt.title(file_b)
    plt.axis("off")

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"Fig.{i}_{j}.png")
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    DIR_A, DIR_B, RESULT_DIR = load_detect_same_image_data(JSON_PATH)
    DIR_A = get_absolute_path_if_not(get_base_dir(__file__), DIR_A)
    DIR_B = get_absolute_path_if_not(get_base_dir(__file__), DIR_B)
    RESULT_DIR = get_absolute_path_if_not(get_base_dir(__file__), RESULT_DIR)
    
    results = compare_all(DIR_A, DIR_B)
    for result in results:
        display(result, RESULT_DIR) 
