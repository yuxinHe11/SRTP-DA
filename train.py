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
prompt_type = 1
print('prompt type: ', prompt_type)


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-cfg', default='./configs/SurgVisDom/SurgVisDom.yaml')
    parser.add_argument('--config', '-%s' % S, default='./configs/SurgVisDom/SurgVisDom_train.yaml')

    # parser.add_argument('--log_time', default='')
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

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

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

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,
                                       pretrain=config.network.init, joint = config.network.joint)  # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    # print('train transforms: {}'.format(transform_train.transforms))
    # print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header,
                                 clip_state_dict,config.data.num_segments, 6)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    # wandb.watch(model)
    # wandb.watch(fusion_model)


    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))




    train_data = SurgVisDom(config.data.train_list, config.data.label_list,  # Todo train_list
                                 num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                                 random_shift=config.data.random_shift, transform=transform_train)
    # train_loader = DataLoader(train_data, batch_size=config.data.batch_size,
    #                           num_workers=config.data.workers, shuffle=True, pin_memory=False,drop_last=True)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size,
                              num_workers=1, shuffle=True, pin_memory=False, drop_last=True)

    val_data = SurgVisDom(config.data.val_list, config.data.label_list,
                               random_shift=False, num_segments=config.data.num_segments,
                               image_tmpl=config.data.image_tmpl, transform=transform_val)
    # val_loader = DataLoader(val_data, batch_size=config.data.batch_size,
    #                         num_workers=config.data.workers, shuffle=False, pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size,
                            num_workers=1, shuffle=False, pin_memory=False, drop_last=False)

    classes, num_text_aug, text_dict = text_prompt(train_data, prompt=prompt_type)
    # classes: [48, 77] = [num_text_aug*num_classes, 77]
    # num_text_aug=16
    # text_dict: 16个key，每个value[3, 77]
    # pdb.set_trace()

    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    # if config.solver.evaluate:
    #     prec1 = validate(start_epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug, logger)
    #     return

    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    best_epoch = -1
    best_list = []
    for epoch in range(start_epoch, config.solver.epochs):
        epoch_loss = 0.0
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk, (images,list_id) in enumerate(tqdm(train_loader)):
            # images: [bs, 48, 224, 224]
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            # images: [bs, num_seg, 3, 224, 224]

            b,t,c,h,w = images.size()
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])
            # texts: [bs, 77]

            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            # images: [512, 3, 224, 224]
            texts = texts.to(device)

            image_embedding = model_image(images)  # [512, 512]
            # pdb.set_trace()
            image_embedding = image_embedding.view(b,t,-1)  # [bs, num_seg, 512]
            image_embedding = fusion_model(image_embedding)  # [bs, 512]

            text_embedding = model_text(texts)  # [bs, 512]

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text, logits_ii, logits_tt = create_logits(image_embedding, text_embedding, logit_scale)
            # [bs,bs]


            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)

            # pdb.set_trace()
            # Todo 相同类别的similarity大，不同类别的小
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            # pdb.set_trace()
            loss_ii = loss_img(logits_ii, ground_truth)
            loss_tt = loss_txt(logits_tt, ground_truth)

            total_loss = (loss_imgs + loss_texts)/2.0 + config.solver.ii_tt * (loss_ii + loss_tt)/2.0

            # wandb.log({"train_total_loss": total_loss})
            # wandb.log({"train_loss_imgs": loss_imgs})
            # wandb.log({"train_loss_texts": loss_texts})
            # wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()
            epoch_loss += total_loss.cpu().data

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print('epoch loss: ', epoch_loss/(kkk+1))
        if epoch % config.logging.eval_freq == 0:  # and epoch>0
           prec1 = validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, logger)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if is_best:
            best_epoch = epoch
            best_saving(working_dir, epoch, model, fusion_model, optimizer)
            best_list.append(epoch)
        print('best epoch: ', best_epoch)
        print('best list: ', best_list)


if __name__ == '__main__':
    main()
