import os
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from train import val_transform
from utils.path_solver import get_base_dir, get_absolute_path_if_not
from utils.settings import load_predict_data, JSON_PATH

def predict(image_path, model, device):
    img = Image.open(image_path).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        
    max_prob = max_prob.item()
    class_id = pred.item()
        
    return class_id, max_prob
    
def load_model(model_path, class_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, class_num)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device

def display(image_path, class_name, class_id, prob):
    img = Image.open(image_path)
    plt.figure(figsize=(10,4))
    plt.imshow(img)
    plt.title(f"Predicted : {class_name}, id : {class_id}, prob: {prob}", fontsize=16)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    TARGET, MODEL_PATH, DATASET_DIR= load_predict_data(JSON_PATH)
    TARGET = get_absolute_path_if_not(get_base_dir(__file__), TARGET)
    DATASET_DIR = get_absolute_path_if_not(get_base_dir(__file__), DATASET_DIR)
    # DATASET_DIRからクラス名を取得
    class_names = sorted(
        [name for name in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, name))]
    )
    model, device = load_model(MODEL_PATH, len(class_names))

    # TARGETディレクトリ内の全ての画像絶対パスを取得
    image_pathes = sorted(
        [os.path.join(TARGET, f) for f in os.listdir(TARGET) if f.lower().endswith(("png", "jpg", "jpeg"))]
    )
    for image_path in tqdm(image_pathes):
        id, prob = predict(image_path, model, device)
        name = class_names[id]
        display(image_path, name, id, prob)

