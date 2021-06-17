from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import pandas as pd
import sys
import random
import torch
import numpy as np
import os
import json
from transformers import set_seed
from tqdm import tqdm
from shutil import copyfile


def create_folds(data, num_splits, seed):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    return data

from augmentation import Augmenter

def update_config(orig_config_path, new_config_path, new_config):
    copyfile(orig_config_path, new_config_path)
    with open(orig_config_path) as json_file:
        config = json.load(json_file)
        
        for k, v in new_config.items():
            config[k] = v
            
    with open(new_config_path, "w") as outfile: 
        json.dump(config, outfile, indent=4)

def augment(name, config):
    num_fold = 5
    seed = 2021
    data_dir = os.path.join('data', 'aug_data')
    
    dataset = pd.read_csv('data/raw/train.csv')
    dataset = create_folds(dataset, num_splits=num_fold, seed=seed)
    
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    
    os.makedirs(os.path.join(data_dir, name), exist_ok=True)
    config_path = os.path.join(data_dir, name, 'augmentation_config.json')
    orig_data_output_path = os.path.join(data_dir, name, 'kfold_orig_data.csv')
    aug_data_output_path = os.path.join(data_dir, name, 'kfold_aug_data.csv')
    
    update_config('augmentation_config.json', config_path, config)
    
    augmenter = Augmenter(config_path)
    if augmenter.config['device'] == 'cpu':
        torch.set_num_threads(32)

    kfold_aug_data = []

    for k in range(num_fold):
        aug_data = augmenter.generate(dataset[dataset['kfold'] == k])
        aug_data['kfold'] = k
        kfold_aug_data.append(aug_data)
        
    kfold_aug_data= pd.concat(kfold_aug_data)
    kfold_aug_data.to_csv(aug_data_output_path, index=False)
    dataset.to_csv(orig_data_output_path, index=False)
    

if __name__ == "__main__":
    names = [
        'bert_p5',
        'bert_p10',
        'bert_p20',
        'bert_p40',
        'bert_p60',
        'roberta_p10',
        'roberta_p20',
        'xlnet_p10',
        'xlnet_p20',
    ]
    configs = [
        {'aug_p': 0.05},
        {'aug_p': 0.1},
        {'aug_p': 0.2},
        {'aug_p': 0.4},
        {'aug_p': 0.6},
        {
            "augs": [
                [
                    "context_word_embs",
                    "roberta-base",
                ]
            ],
            'aug_p': 0.1
        },
        {
            "augs": [
                [
                    "context_word_embs",
                    "roberta-base",
                ]
            ],
            'aug_p': 0.2
        },
        {
            "augs": [
                [
                    "context_word_embs",
                    "xlnet-base-cased"
                ]
            ],
            'aug_p': 0.1
        },
        {
            "augs": [
                [
                    "context_word_embs",
                    "xlnet-base-cased"
                ]
            ],
            'aug_p': 0.2
        }
    ]
    
    for name, config in tqdm(zip(names, configs)):
        augment(name, config)
    
