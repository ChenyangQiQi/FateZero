import torch
import clip
from PIL import Image
from glob import glob
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def crop_read_image_path(image_path):
    origin_image = Image.open(image_path)
    w, h = origin_image.size
    if h > w:
        origin_image = origin_image.crop((0, h-w, w, h))
    return origin_image


def edit_success(image_path, source_prompt,target_prompt):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)

    text = clip.tokenize([source_prompt, target_prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  
    return probs[0,1] >= probs[0,0], image_features


def folder_success(folder, source_prompt, target_prompt):
    print(folder)
    file_list = sorted(glob(folder+'/*png'))
    normalized_feature_list = []
    print(file_list)
    count = 0.0
    for f_path in file_list:        
        success, image_feature = edit_success(f_path, source_prompt,target_prompt)
        if success: count +=1.0
        normalized_feature_list.append(image_feature/torch.sqrt(torch.sum(image_feature**2, axis=1, keepdims=True)))
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list)-1):
        sim_i = torch.sum(normalized_feature_list[i]*normalized_feature_list[i+1], axis=1)
        frame_const_list.append( sim_i )
        frame_const_list_sum += sim_i
    frame_const_list_avg = frame_const_list_sum/(len(normalized_feature_list)-1)
    print(f'average temporal frame consistency: {frame_const_list_avg}')

    return count/len(file_list), frame_const_list_sum/(len(normalized_feature_list)-1)

from omegaconf import OmegaConf
config_yaml = 'CLIP/bench_clean_prompt.yaml'
Omegadict = OmegaConf.load(config_yaml)
print(Omegadict)
import os


dict_folder = './baselines_results/ours'
sub_folder_list = sorted(glob(f'{dict_folder}/*'))

folder_success_rate_list = []
folder_temp_const_list = []

for k in sub_folder_list:
    print(k)
    v= Omegadict[os.path.basename(k)]
    folder_success_rate, folder_temp_const = folder_success(k, v['source'], v['target'])
    print(f'folder_success_rate {folder_success_rate}')
    print(f'folder_temporal_consistency {folder_temp_const}')
    folder_success_rate_list.append(folder_success_rate)
    folder_temp_const_list.append(folder_temp_const.detach().cpu().numpy()[0])

print('folder_success_rate list :')
print(folder_success_rate_list)    

print('folder_temporal_consistency list :')
print(folder_temp_const_list)    

import numpy as np
dataset_average_rate = (np.array(folder_success_rate_list)).mean()
dataset_average_tempconst = np.array(folder_temp_const_list).mean()

print(f'dataset_average_rate {dataset_average_rate}')
print(f'dataset_average_tempconst {dataset_average_tempconst}')