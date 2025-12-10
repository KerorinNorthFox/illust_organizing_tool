import os
import torch
import torch.nn.functional as F
from torchvision import models
from train import val_transform
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def predict(image_path, model, class_names, device):
    img = Image.open(image_path).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        
    max_prob = max_prob.item()
    class_id = pred.item()
    if max_prob < 0.7:
        return "other", -1, max_prob
        
    return class_names[class_id], class_id, max_prob
    
def load_model(model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
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
    DIR = "C:\\Images"
    MODEL = "model.pth"
    
    image_pathes = sorted(
        [os.path.join(DIR, f) for f in os.listdir(DIR) if f.lower().endswith(("png", "jpg", "jpeg"))]
    )
    class_names = ["hoshino", "kisaki", "mari", "seia"]
    model, device = load_model(MODEL, class_names)

    for image_path in tqdm(image_pathes):
        name, id, prob = predict(image_path, model, class_names, device)
        display(image_path, name, id, prob)

