import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.path_solver import get_base_dir, get_absolute_path_if_not
from utils.settings import load_train_data, JSON_PATH

DATASET_DIR, _, _, _ = load_train_data(JSON_PATH)
DATASET_DIR = get_absolute_path_if_not(get_base_dir(__file__), DATASET_DIR)
BATCH_SIZE = 16

### 画像の前処理 ###
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
    ),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### データセット読み込み ###
train_dataset_dir = os.path.join(DATASET_DIR, "training_set")
train_dataset = datasets.ImageFolder(train_dataset_dir, transform=train_transform)
num_classes = len(train_dataset.classes)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Dataset dir:", train_dataset_dir)
print("クラス数:", train_dataset.classes)

label_map = { 0: "cat", 1: "dog" }

def display_raw_pca(): # 画素値によるPCAをグラフに表示
    # Load images data
    data, labels = [], []
    for images, labs in train_loader:
        gray = images.mean(dim=1)
        gray = gray.view(gray.size(0), -1)
        gray = gray.cpu().numpy()
        data.append(gray)
        labels.append(labs)
    data = np.vstack(data)
    labels = np.concatenate(labels)
    print("Data shape:", data.shape)
    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)
    class0_pc = []
    class1_pc = []
    # Display PCA
    for cls in np.unique(labels):
        idx = labels == cls
        if cls == 0:
            class0_pc.append(proj[idx, :])
        else:
            class1_pc.append(proj[idx, :])
        plt.scatter(proj[idx, 0], proj[idx, 1], s=10, label=label_map[cls], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
    
    return class0_pc, class1_pc, labels, pca

def display_resnet_pca(): # ResNetの特徴量によるPCAをグラフに表示
    ### Define ResNet50 model ###
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity() # 最終層は特徴ベクトル抽出用に入力をそのまま通す
    model = model.to(device)
    model.eval()
    # Sample features
    features, labels = [], []
    with torch.no_grad():
        for images, labs in tqdm(train_loader):
            images = images.to(device)
            feat = model(images)
            if feat.ndim > 2: feat = torch.nn.Flatten(feat, 1)
            features.append(feat.cpu().numpy())
            labels.append(labs.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    print("feature shape: ", features.shape)
    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    class0_pc, class1_pc = [], []
    # Display PCA
    for cls in np.unique(labels):
        idx = labels == cls
        if cls == 0: class0_pc.append(proj[idx, :]) 
        else: class1_pc.append(proj[idx, :])
        plt.scatter(proj[idx, 0], proj[idx, 1], label=str(cls), s=6, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

    return class0_pc, class1_pc, labels, pca, model

def predict(model, image_path, pca): # 画素値のPCAを用いた線形回帰モデルでクラス予測
    # 画像変換
    img = Image.open(image_path).convert("RGB")
    img_tensor = train_transform(img).unsqueeze(0)
    gray = img_tensor.mean(dim=1)
    gray = gray.view(gray.size(0), -1)
    data = gray.cpu().numpy()
    # 未知画像のpca抽出
    proj = pca.transform(data)
    pc = np.array([[proj[0, 0], proj[0, 1]]])
    # 予測
    pred_label = model.predict(pc)
    pred_prob = model.predict_proba(pc)
    
    print(f"image_path: {image_path}, result: {label_map[int(pred_label[0])]}, probability: {pred_prob}")
    return int(pred_label[0]), pred_prob

def predict_resnet(model, image_path, pca, resnet): # ResNetの特徴量のPCAを用いた線形回帰モデルでクラス予測
    # 画像変換
    img = Image.open(image_path).convert("RGB")
    img = train_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img)
    feat = feat.cpu().numpy()
    # 未知画像のpca抽出
    proj = pca.transform(feat)
    # 予測
    pred_label = model.predict(proj)
    pred_prob = model.predict_proba(proj)
    
    print(f"image_path: {image_path}, result: {label_map[int(pred_label[0])]}, probability: {pred_prob}")
    return int(pred_label[0]), pred_prob

def create_model(class0_pc_list, class1_pc_list): # PCA結果から線形回帰モデル作成
    # データ成形
    X_class0 = np.vstack(class0_pc_list)
    X_class1 = np.vstack(class1_pc_list)
    X_train = np.vstack([X_class0, X_class1])
    y_train = np.array([0]*len(X_class0) + [1]*len(X_class1))
    # モデル作成
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # 予測データ準備
    cat_image_dir = os.path.join(DATASET_DIR, "test_set/cats")
    dog_image_dir = os.path.join(DATASET_DIR, "test_set/dogs")
    cat_image_path_list = [os.path.join(cat_image_dir, file) for file in os.listdir(cat_image_dir) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    dog_image_path_list = [os.path.join(dog_image_dir, file) for file in os.listdir(dog_image_dir) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    # PCA表示＆モデル作成
    raw_class0_pc_list, raw_class1_pc_list, raw_labels, pca = display_raw_pca()
    model = create_model(raw_class0_pc_list, raw_class1_pc_list)
    rn_class0_pc_list, rn_class1_pc_list, rn_labels, pca_rn, resnet_rn = proj2 = display_resnet_pca()
    model_rn = create_model(rn_class0_pc_list, rn_class1_pc_list)
    
    # 猫の画像分類
    cat_raw_prob_list, cat_rn_prob_list = [], []
    cat_raw_legit_list, cat_rn_legit_list = [], []
    for cat_path in cat_image_path_list:
        pred_label_raw, pred_prob_raw = predict(model, cat_path, pca)
        cat_raw_prob_list.append(pred_prob_raw)
        if pred_label_raw == 0: cat_raw_legit_list.append(pred_label_raw)
        
        pred_label_rn, pred_prob_rn = predict_resnet(model_rn, cat_path, pca_rn, resnet_rn)
        cat_rn_prob_list.append(pred_prob_rn)
        if pred_label_rn == 0: cat_rn_legit_list.append(pred_label_rn)
        
    # 犬の画像分類
    dog_raw_prob_list, dog_rn_prob_list = [], []
    dog_raw_legit_list, dog_rn_legit_list = [], []
    for dog_path in dog_image_path_list:
        pred_label_raw, pred_prob_raw = predict(model, dog_path, pca)
        dog_raw_prob_list.append(pred_prob_raw)
        if pred_label_raw == 1: dog_raw_legit_list.append(pred_label_raw)
        
        pred_label_rn, pred_prob_rn = predict_resnet(model_rn, dog_path, pca_rn, resnet_rn)
        dog_rn_prob_list.append(pred_prob_rn)
        if pred_label_rn == 1: dog_rn_legit_list.append(pred_label_rn)
        
    # 結果表示
    print("画素値によるpcaの線形回帰予測結果：")
    print(f"cat -> 平均精度: {sum(cat_raw_prob_list) / len(cat_image_path_list)}, 正答率: {len(cat_raw_legit_list) / len(cat_image_path_list)}")
    print(f"dog -> 平均精度: {sum(dog_raw_prob_list) / len(dog_image_path_list)}, 正答率: {len(dog_raw_legit_list) / len(dog_image_path_list)}")
    print("\nResNetによるpcaの線形回帰予測結果：")
    print(f"cat -> 平均精度: {sum(cat_rn_prob_list) / len(cat_image_path_list)}, 正答率: {len(cat_rn_legit_list) / len(cat_image_path_list)}")
    print(f"dog -> 平均精度: {sum(dog_rn_prob_list) / len(dog_image_path_list)}, 正答率: {len(dog_rn_legit_list) / len(dog_image_path_list)}")
