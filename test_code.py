import os
import pdb
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import torch.nn as nn
from datasets import SurgVisDom
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from test import validate, validate_val
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
from sklearn.model_selection import StratifiedKFold
import logging

S = 'cfg'
global args, best_prec1
global global_step
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-%s' % S, default='./configs/SurgVisDom/SurgVisDom_train.yaml')

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], time_now)

    # wandb.init(project=config['network']['type'],name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

transform_train = get_augmentation(True, config)

train_data = SurgVisDom(config.data.train_list, config.data.label_list,  # Todo train_list
                                 num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                                 random_shift=config.data.random_shift, transform=transform_train)

a,b,c = text_prompt(train_data)

print(a)