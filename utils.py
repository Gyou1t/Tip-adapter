from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict, Counter
import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            texts = clip.tokenize(classname).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


# tip cache
def build_cache_model(load_cache, train_loader_cache, model):
    if not load_cache:    
        max_samples = 200  # 每个类别最多选100张图片
        # 用字典存储每个类别的特征列表，键为类别，值为对应的特征列表（保存在CPU上）
        class_features = defaultdict(list)
        # model.load_state_dict(torch.load("/home/bai/下载/Dassl.pytorch-master/clip-Medic_epoch_34.pth", map_location="cuda"))
        # model.load_state_dict(torch.load("/opt/picture/clip_epoch_198.pth", map_location="cuda"))
        # Data augmentation：这里用10个增强轮次
        for augment_idx in range(10):
            print('Augment Epoch: {:} / {:}'.format(augment_idx + 1, 10))
            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    # 提取特征（在GPU上计算，然后转到CPU保存以减少内存占用）
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # 遍历每个batch中的样本
                    for feat, target in zip(image_features, targets):
                        cls = target.item()
                        if cls == 0:
                            sampleM = 500
                        elif cls in [1, 2, 3]:
                            sampleM = 500
                        elif cls == 4:
                            sampleM = 500
                        else:
                            sampleM = max_samples  # 默认值
                        if len(class_features[cls]) < sampleM:
                            class_features[cls].append(feat.cpu())
                            
        # 将每个类别的特征拼接起来
        all_features = []
        all_targets = []
        for cls, feats in class_features.items():
            if len(feats) > 0:
                feats_tensor = torch.stack(feats, dim=0)  # (n, feature_dim)
                all_features.append(feats_tensor)
                all_targets.extend([cls] * feats_tensor.size(0))
        
        # 拼接所有类别的特征，得到 (total_samples, feature_dim)
        cache_keys = torch.cat(all_features, dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        # 转换为 (feature_dim, total_samples)
        cache_keys = cache_keys.transpose(0, 1)
        
        # 对 all_targets 进行 one-hot 编码，假设类别从0开始
        all_targets_tensor = torch.tensor(all_targets)
        num_classes = all_targets_tensor.max().item() + 1
        cache_values = F.one_hot(all_targets_tensor, num_classes=num_classes).float()
        
        # 统计各类别的样本数量及占比
        counter = Counter(all_targets)
        total_count = sum(counter.values())
        print("类别分布:")
        for cls, count in sorted(counter.items()):
            print(f"类别 {cls}: {count} ({count/total_count:.2%})")
        
        # 保存缓存数据
        torch.save(cache_keys, "/opt/picture/keys_200shots.pt")
        torch.save(cache_values, "/opt/picture/values_200shots.pt")
    else:
        cache_keys = torch.load("/opt/picture/keys_200shots.pt")
        cache_values = torch.load("/opt/picture/values_200shots.pt")
    
    return cache_keys, cache_values


def search_hp( cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    
    beta_list = [i * ([12,5][0] - 0.1) /[200, 20][0] + 0.1 for i in range([200, 20][0])]
    alpha_list = [i * ([12,5][1] - 0.1) / [200, 20][1] + 0.1 for i in range([200, 20][1])]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    for beta in beta_list:
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, labels)
        
            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha
