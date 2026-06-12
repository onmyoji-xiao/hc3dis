import os

import open_clip
from open_clip import tokenizer
import torch
import numpy as np
from evaluation.constants import MATTERPORT_LABELS, SCANNET_LABELS, SCANNETPP_LABELS


def extract_text_feature(save_path, descriptions, device):
    text_tokens = tokenizer.tokenize(descriptions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    text_features_dict = {}
    for i, description in enumerate(descriptions):
        text_features_dict[description] = text_features[i]

    np.save(save_path, text_features_dict)


device = torch.device('cuda:1')
model, _, _ = open_clip.create_model_and_transforms("ViT-H-14",
                                                    pretrained='../pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')
model = model.to(device)
model.eval()

save_dir = '../data/text_features'
os.makedirs(save_dir, exist_ok=True)
# extract_text_feature(os.path.join(save_dir, 'scannet.npy'), SCANNET_LABELS, device)
extract_text_feature(os.path.join(save_dir, 'scannetpp.npy'), SCANNETPP_LABELS, device)
