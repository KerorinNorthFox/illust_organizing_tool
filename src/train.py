import os
import time
import datetime
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from torchinfo import summary
from tqdm import tqdm

from utils.path_solver import get_base_dir, get_absolute_path_if_not
from utils.logger import export_train_plot, export_train_logs
from utils.settings import load_train_data, JSON_PATH

DATASET_DIR, SAVE_DIR, MODEL_NAME, EPOCHS = load_train_data(JSON_PATH)
DATASET_DIR = get_absolute_path_if_not(get_base_dir(__file__), DATASET_DIR)
VAL_RATIO = 0.2
BATCH_SIZE = 32

### 画像の前処理 ###
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224, 224)),
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

total_size = len(base_dataset)
val_size = int(total_size * VAL_RATIO)
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

### 学習, 評価関数 ###
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # モデルを学習モードに設定
    train_loss = 0.0
    train_acc = 0.0
    scaler = torch.GradScaler()

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # 勾配初期化

        with torch.autocast(str(device)):
            outputs = model(images)
            loss = criterion(outputs, labels) # 損失の計算
        train_loss += loss.item() * images.size(0) # 損失蓄積
        acc = (outputs.max(1)[1] == labels).sum() # 精度計算
        train_acc += acc.item()
        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()
        
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
    print("Dataset dir:", DATASET_DIR)
    print("クラス数:", base_dataset.classes)
    
    ### モデル定義 ###
    weights = models.ResNet50_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model_info = summary(model, input_size=(BATCH_SIZE, 3, 224, 224))
    print(model_info)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 学習
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_time_list = []
    val_time_list = []
    epochs = EPOCHS
    for epoch in range(epochs):
        print(f"Epochs: {epoch+1}")
        
        start_train = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        end_train = time.perf_counter()

        start_val = time.perf_counter()
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        end_val = time.perf_counter()

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        train_time_list.append((end_train-start_train)/60)
        val_time_list.append((end_val-start_val)/60)

        print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # モデルの保存
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    SAVE_DIR = SAVE_DIR.replace("{datetime}", now_str)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
    print("Saved model.")

    # データのエクスポート
    epoch_list = list(range(epochs))
    export_train_plot(epoch_list, train_loss_list, val_loss_list, "loss", "train / val loss", os.path.join(SAVE_DIR, "loss.png"))
    export_train_plot(epoch_list, train_acc_list, val_acc_list, "acc", "train / val acc", os.path.join(SAVE_DIR, "acc.png"))
    export_train_logs(SAVE_DIR, DATASET_DIR, VAL_RATIO, total_size, train_size, val_size, model_info, BATCH_SIZE, base_dataset.classes, epochs, train_loss_list, train_acc_list, val_loss_list, val_acc_list, train_time_list, val_time_list)
