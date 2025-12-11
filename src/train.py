import os
import pathlib
import datetime
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from settings import load_train_data, JSON_PATH

DATASET_DIR, MODEL_PATH_SAVE, EPOCHS = load_train_data(JSON_PATH)
if not pathlib.Path(DATASET_DIR).is_absolute():
   DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), DATASET_DIR)
print("Dataset dir:", DATASET_DIR)

### 画像の前処理 ###
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
    ),
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
    ),
])

### データセット読み込み ###
base_dataset = datasets.ImageFolder(DATASET_DIR, transform=None)
num_classes = len(base_dataset.classes)
print("クラス数:", base_dataset.classes)

val_ratio = 0.2
total_size = len(base_dataset)
val_size = int(total_size * val_ratio)
train_size = total_size - val_size

train_indices, val_indices = random_split(
    range(total_size),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset_full = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
val_dataset_full = datasets.ImageFolder(DATASET_DIR, transform=val_transform)

train_dataset = Subset(train_dataset_full, train_indices.indices)
val_dataset = Subset(val_dataset_full, val_indices.indices)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

### 学習, 評価関数 ###
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # モデルを学習モードに設定
    train_loss = 0.0
    train_acc = 0.0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # 勾配初期化

        outputs = model(images)
        loss = criterion(outputs, labels) # 損失の計算
        train_loss += loss.item() * images.size(0) # 損失蓄積
        acc = (outputs.max(1)[1] == labels).sum() # 精度計算
        train_acc += acc.item()
        loss.backward()
        optimizer.step()
        
    avg_train_loss = train_loss / len(dataloader.dataset) # 損失平均
    avg_train_acc = train_acc / len(dataloader.dataset) # 精度平均
    return avg_train_loss, avg_train_acc

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()
        
    avg_val_loss = val_loss / len(dataloader.dataset)
    avg_val_acc = val_acc / len(dataloader.dataset)
    return avg_val_loss, avg_val_acc


if __name__ == "__main__":
    ### モデル定義 ###
    weights = models.ResNet50_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    epochs = EPOCHS
    for epoch in range(epochs):
        print(f"Epochs: {epoch+1}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    now = datetime.datetime.now()
    model_name = MODEL_PATH_SAVE.replace("{datetime}", now.strftime("%Y-%m-%d-%H-%M-%S"))
    torch.save(model.state_dict(), model_name)
    print("Saved model.")
