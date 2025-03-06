import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import clip
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
import numpy
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import pdb
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import logging
'''
2025.3.6
李安
阅读、注释并尝试修改测试代码。希望跑SDA-CLIP作者提供的权重时可以达到论文中的效果。
'''



class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def val_metrics(pred, logger):
    test_num_each = [5464, 5373, 27014, 4239, 3936, 6258, 10474, 6273,
                     10512, 6667, 22131, 4661, 8855, 14047, 28896, 4209]
    # 测试集共16（0-15）个video，test_num_each是每个video的帧数（然而第3个元素标错了，应该是27014，此文件中已改正）.
    test_num_snippet = [43, 42, 212, 34, 31, 49, 82, 50, 83, 53, 173, 37, 70, 110, 226, 33]
    # 似乎test_num_snippet的每个元素的值等于帧数/128.

    mean_weighted_f1 = 0.0
    mean_unweighted_f1 = 0.0
    mean_global_f1 = 0.0
    mean_balanced_acc = 0.0
    each_wf1 = []
    each_unf1 = []
    each_gf1 = []
    each_bacc = []

    test_labels_pth = '/mnt/sdc/heyuxin/surgvisdom_image/test_labels/'

    for i in range(16):
        predi = pred[sum(test_num_snippet[:i]): sum(test_num_snippet[:i+1])]
        # sum(test_num_snippet[:i])是test_num_snippet中前i个元素的和
        # sum(test_num_snippet[:i+1])是test_num_snippet中前i+1个元素的和
        predi = [p for p in predi for _ in range(128)]
        predi = predi[:test_num_each[i]]
        # predi = pred[sum(val_num_each[:i]): sum(val_num_each[:i+1])]

        tl_pth = test_labels_pth + '/test_video_' + str(i).zfill(4) + '.csv'
        ls = np.array(pd.read_csv(tl_pth, usecols=['frame_label']))
        # ls = np.nan_to_num(ls, nan=-1)  # 将 NaN 替换为 -1

        label = []
        predict = []
        for idx, l in enumerate(ls):
            if not np.isnan(l):
                if idx < len(predi):
                    label.append(int(l))
                    predict.append(predi[idx])
                else:
                    logger.warning(f"Video {i}: 预测长度不足，标签索引 {idx} 被忽略")
        # for idx, l in enumerate(ls):
        #     if not np.isnan(l):
        #         label.append(int(l))
        #         # print('length of ls:', len(ls))
        #         # print('length of label:', len(label))
        #         # print('length of predi:', len(predi))
        #         # breakpoint()
        #         predict.append(predi[idx])

        # pdb.set_trace()
        mean_weighted_f1 += f1_score(label, predict, average='weighted')/16.0
        mean_unweighted_f1 += f1_score(label, predict, average='macro') / 16.0
        mean_global_f1 += f1_score(label, predict, average='micro') / 16.0
        mean_balanced_acc += balanced_accuracy_score(label, predict) / 16.0

        each_wf1.append(f1_score(label, predict, average='weighted'))
        each_unf1.append(f1_score(label, predict, average='macro'))
        each_gf1.append(f1_score(label, predict, average='micro'))
        each_bacc.append(balanced_accuracy_score(label, predict))
        # print('video: ', i, 'label: ', label, 'predict: ', predict)

    logger.info('wf1: {}'.format(each_wf1))
    logger.info('unf1:{}'.format(each_unf1))
    logger.info('gf1:{}'.format(each_gf1))
    logger.info('bacc:{}'.format(each_bacc))


    return mean_weighted_f1, mean_unweighted_f1, mean_global_f1, mean_balanced_acc


def validate_val(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    predict_list = []
    label_list = []
    label2 = []
    pred2 = []

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)  # (bs*num_classes, 512)
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            # image: (bs, 24, 224, 224)
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            # image: (16, 8, 3, 224, 224)
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)  # (bs, 512)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1)
            # pdb.set_trace()

            similarity = similarity.softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            # values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b

            # print(indices_1)
            # print(class_id)
            # pdb.set_trace()

            for i in range(b):
                if values_1[i] < 0.5:
                    indices_1[i] = -1

                # pdb.set_trace()

                label_list.append(int(class_id[i].cpu().numpy()))
                predict_list.append(indices_1[i].cpu().numpy()[0])

                # if indices_1[i] == class_id[i]:
                #     corr_1 += 1
                # if class_id[i] in indices_5[i]:
                #     corr_5 += 1
            # pdb.set_trace()

    # f1score = f1_score(label2, pred2, average='weighted')
    # acc = accuracy_score(label2, pred2)
    # pdb.set_trace()
    bacc = balanced_accuracy_score(label_list, predict_list)
    print('Epoch: [{}/{}]: bacc:{}'.format(epoch, config.solver.epochs, bacc))
    return bacc


def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, logger):
    '''
    epoch:int 0
    val_loader:验证集
    classes:见Text_Prompt.py
    device:'cuda'
    num_text_aug:见Text_Prompt.py
    '''
    model.eval()
    fusion_model.eval() # 模型切换到评估模式
    num = 0
    corr_1 = 0
    corr_5 = 0

    predict_list = []
    label_list = []
    label2 = []
    pred2 = []

    with torch.no_grad():
        # 关闭PyTorch自动梯度计算
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)  # (bs*num_classes, 512)
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            # image: (bs, 24, 224, 224)
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            # image: (16, 8, 3, 224, 224)
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)  # (bs, 512)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1)
            # pdb.set_trace()

            similarity = similarity.softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            # values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b

            print(indices_1)
            print(class_id)
            # pdb.set_trace()

            for i in range(b):
                # if values_1[i] < 0.5:
                #     indices_1[i] = -1

                # pdb.set_trace()

                # label_list.append(int(class_id[i].cpu().numpy()))
                predict_list.append(indices_1[i].cpu().numpy()[0])

                # if indices_1[i] == class_id[i]:
                #     corr_1 += 1
                # if class_id[i] in indices_5[i]:
                #     corr_5 += 1
    # pdb.set_trace()

    # f1score = f1_score(label2, pred2, average='weighted')
    # acc = accuracy_score(label2, pred2)

    wf1, unf1, gf1, bacc  = val_metrics(predict_list, logger)
    # top1 = f1score
    # top5 = float(corr_5) / num * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    # print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    logger.info('Epoch: [{}/{}]: wf1:{:.3f} unf1:{:.3f} gf1:{:.3f} bacc:{:.3f}'.format(epoch, config.solver.epochs, wf1, unf1, gf1, bacc))
    return wf1


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/SurgVisDom/SurgVisDom_test.yaml')
    parser.add_argument('--log_time', default='20250305-2221')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)

    os.environ["WANDB_API_KEY"] = '3aa1219568e63fe92521fc3220a6a508dc21b993'  # 将引号内的+替换成自己在wandb上的一串值
    os.environ["WANDB_MODE"] = "offline"  # 离线  （此行代码不用修改）

    wandb.init(project=config['network']['type'], name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)
    print(transform_val)
    breakpoint()


    # text encoder, image encoder和fusion_model
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    # 移动到GPU上
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    # 加载测试集
    val_data = SurgVisDom(config.data.test_list, config.data.label_list, num_segments=config.data.num_segments,
                          image_tmpl=config.data.image_tmpl,
                          transform=transform_val, random_shift=config.random_shift, test_mode= True)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=1, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)  
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch # 默认为0

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            # 加载预训练的模型权重用于测试
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(data=val_data, prompt=4)

    def getLogger():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")

        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)

        logger.addHandler(sHandler)

        pth = os.path.join(working_dir, working_dir.split('/')[-1]+'.txt')

        fHandler = logging.FileHandler(pth, mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)
        return logger

    logger = getLogger()

    best_prec1 = 0.0
    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, logger)

if __name__ == '__main__':
    main()
