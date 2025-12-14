import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from train import val_transform
from model_container import ModelContainer
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

def display(image_path, class_name, class_id, prob):
    img = Image.open(image_path)
    plt.figure(figsize=(10,4))
    plt.imshow(img)
    plt.title(f"Predicted : {class_name}, id : {class_id}, prob: {prob}", fontsize=16)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    target, dataset_dir, model_path, model_type = load_predict_data(JSON_PATH)
    target = get_absolute_path_if_not(get_base_dir(__file__), target)
    dataset_dir = get_absolute_path_if_not(get_base_dir(__file__), dataset_dir)
    # DATASET_DIRからクラス名を取得
    class_names = sorted(
        [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    )
    model, device = ModelContainer.select(model_type, is_weights=False, num_classes=len(class_names), model_path=model_path)
    model.eval()

    # TARGETディレクトリ内の全ての画像絶対パスを取得
    image_pathes = sorted(
        [os.path.join(target, f) for f in os.listdir(target) if f.lower().endswith(("png", "jpg", "jpeg"))]
    )
    for image_path in tqdm(image_pathes):
        id, prob = predict(image_path, model, device)
        name = class_names[id]
        display(image_path, name, id, prob)

