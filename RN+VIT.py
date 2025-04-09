import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score
import numpy as np
#0.5 2e-4 2e-3 0.2 0.8 85.56%
#0.6 2e-4 2e-3 0.2 0.8 78%
#0.5 2e-4 2e-3 0.2 0.8 500 86.38%


device = "cuda" if torch.cuda.is_available() else "cpu"
#加载CLIP
model, preprocess = clip.load("ViT-B/32", device=device)
#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
model.eval()  
model_RN, preprocess = clip.load("RN101", device=device)
model_RN.eval()  
# 读 给我读！
csv_path = "/opt/picture/aptos2019-blindness-detection/train.csv"  
df = pd.read_csv(csv_path)
num_rows = df.shape[0]
# 特征文本描述
class_labels = [
    "A clear retinal fundus image with no signs of diabetic retinopathy.",
    "A retinal fundus image with mild diabetic retinopathy, featuring a few scattered microaneurysms.",
    "A retinal fundus image with moderate diabetic retinopathy, showing visible microaneurysms and hemorrhages.",
    "A retinal fundus image with severe diabetic retinopathy, exhibiting multiple hemorrhages, venous beading, and microvascular abnormalities.",
    "A retinal fundus image with proliferative diabetic retinopathy, characterized by neovascularization and extensive retinal damage."
]

# 图特征+标签
images= []
labels = []
class MedicalTransform:
    def __init__(self):
        self.base_transform = preprocess
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3)
        ])
    
    def __call__(self, img):
        return self.base_transform(self.aug_transform(img))

# 在数据加载时应用增强
medical_preprocess = MedicalTransform()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Image Features"):
    img_name = row["id_code"]
    true_label = row["diagnosis"]
    
    img_path = f"/opt/picture/aptos2019-blindness-detection/train_images/{img_name}.png"
    # image = medical_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    images.append(image)
    labels.append(true_label)

# 转换为 Tensor
image_features_cache = torch.cat(images, dim=0)  # (N, D)
labels_cache = torch.tensor(labels, device=device)  # (N,)
dataset_cache = TensorDataset(image_features_cache, labels_cache)

# 分三个
total_size = len(dataset_cache)
train_size = int(0.8 * total_size)   # 80% 训练集
val_size = int(0.1 * total_size)     # 10% 验证集
test_size = total_size - train_size - val_size  # 剩余 10% 作为测试集
train_dataset, val_dataset,test_dataset = random_split(dataset_cache, [train_size,val_size ,test_size])

# 创建 DataLoader
train_loader_cache = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_loader_cache = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=False)
test_loader_cache = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

# Tip-adapter
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
class TipAdapter(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.fc1 = nn.Linear(feature_dim, feature_dim * 2)
        self.bn1 = nn.BatchNorm1d(feature_dim * 2)
        self.fc2 = nn.Linear(feature_dim * 2, feature_dim // 2)
        self.bn2 = nn.BatchNorm1d(feature_dim // 2)
        self.fc3 = nn.Linear(feature_dim // 2, num_classes)
        self.relu = nn.GELU()  
        self.dropout = nn.Dropout(0.5)
        self.residual_proj = nn.Linear(feature_dim, feature_dim*2)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x1, x2):
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        fused = alpha * x1 + beta * x2
        residual = self.residual_proj(x1)
        x = self.relu(self.bn1(self.fc1(fused)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x + residual)))  # 残差连接
        x = self.dropout(x)
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_keys, cache_values = build_cache_model(True, train_loader_cache, model)
cache_keys=cache_keys.to(device)
cache_values=cache_values.to(device)
adapter = TipAdapter(cache_keys.shape[0], cache_keys.shape[1]).to(device)
clip_weights = clip_classifier(class_labels, model).float().to(device)

optimizer = optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=2e-3)
num_epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
# EMA配置
ema = EMA(adapter, 0.999)
ema.register()

# 损失函数
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

best_acc = 0.0
lambda_base = 0.2
beta_base = 0.8

for epoch in range(num_epochs):
    # 动态调整参数
    current_lambda = lambda_base * (0.95 ** (epoch // 30))
    current_beta = beta_base * (1.05 ** (epoch // 20))
    adapter.train()
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(train_loader_cache, desc=f"Epoch {epoch+1}"):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            clip_feat = model.encode_image(images)
            clip_feat = F.normalize(clip_feat, dim=-1)
            rn_feat = model_RN.encode_image(images)
            rn_feat = F.normalize(rn_feat, dim=-1)
        affinity = adapter(clip_feat.float(), rn_feat.float())
        cache_logits = ((-1) * (current_beta - current_beta * affinity)).exp() @ cache_values
        clip_logits = 100.0 * clip_feat.float() @ clip_weights
        tip_logits = (1 - current_lambda) * clip_logits + current_lambda * cache_logits
        loss = criterion(tip_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        preds = tip_logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    scheduler.step()
    adapter.eval()
    val_correct = 0
    val_total = 0
    ema.apply_shadow() # 使用EMA权重
    with torch.no_grad(): 
        for images, targets in val_loader_cache:
            images, targets = images.to(device), targets.to(device)
            
            clip_feat = model.encode_image(images)
            clip_feat = F.normalize(clip_feat, dim=-1)
            rn_feat = model_RN.encode_image(images)
            rn_feat = F.normalize(rn_feat, dim=-1)
            affinity = adapter(clip_feat.float(), rn_feat.float())
            cache_logits = ((-1) * (current_beta - current_beta * affinity)).exp() @ cache_values
            clip_logits = 100.0 * clip_feat.float() @ clip_weights
            tip_logits = (1 - current_lambda) * clip_logits + current_lambda * cache_logits
            preds = tip_logits.argmax(dim=1)
            val_correct += (preds == targets).sum().item()
            val_total += targets.size(0)

    # 保存最佳模型
    val_acc = val_correct / val_total
    ema.restore()
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model': adapter.state_dict(),
            'best_acc': best_acc
        }, 'best_model.pth')
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Acc: {total_correct/total_samples:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Current Lambda: {current_lambda:.3f} | Current Beta: {current_beta:.3f}")
    print(f"Best Val Acc: {best_acc:.4f}\n")

checkpoint = torch.load('best_model.pth')
adapter.load_state_dict(checkpoint['model'])
adapter.eval()

all_preds = []
all_targets = []
all_probs = [] 

with torch.no_grad():
    for images, targets in test_loader_cache:
        images, targets = images.to(device), targets.to(device)
        
        clip_feat = model.encode_image(images)
        clip_feat = F.normalize(clip_feat, dim=-1)
        rn_feat = model_RN.encode_image(images)
        rn_feat = F.normalize(rn_feat, dim=-1)
        
        affinity = adapter(clip_feat.float(), rn_feat.float())
        cache_logits = ((-1) * (current_beta - current_beta * affinity)).exp() @ cache_values
        clip_logits = 100.0 * clip_feat.float() @ clip_weights
        tip_logits = (1 - current_lambda) * clip_logits + current_lambda * cache_logits
        probs = F.softmax(tip_logits, dim=1)
        preds = tip_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# 计算准确率
accuracy = np.mean(all_preds == all_targets)
f1 = f1_score(all_targets, all_preds, average='weighted')
try:
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
except Exception as e:
    print("AUC计算出错：", e)
    auc = None
kappa = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
print(f"测试集准确率: {accuracy * 100:.2f}%")
print(f"测试集 F1-score: {f1:.4f}")
if auc is not None:
    print(f"测试集 AUC: {auc:.4f}")
else:
    print("测试集 AUC计算失败")
print(f"测试集 Cohen's Kappa: {kappa:.4f}")


# 86.38
# 86.21
