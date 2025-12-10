import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from train import val_transform
from tqdm import tqdm

DATASET_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATASET_PATH = os.path.join(DATASET_DIR, "test_set")
MODEL = "model.pth"

test_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def predict(model, dataloader, device):
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            acc = (outputs.max(1)[1] == labels).sum()
            test_acc += acc.item()
            
    avg_test_acc = test_acc / len(dataloader.dataset)
    return avg_test_acc

if __name__ == "__main__":
    num_classes = len(test_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model = model.to(device)

    test_acc = predict(model, test_loader, device)
    print("テストデータに対する精度:", test_acc*100, "%")
