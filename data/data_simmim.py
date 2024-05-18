# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ExpandingMaskGenerator:
    def __init__(self, input_size=196, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        """Note, here we relay on the image to be squares. for different aspact ratios 
        indexing logic might need to be changed.
        assums 0 is masked 1 is unmasked """ 
        assert np.sqrt(input_size)**2 == input_size, "the input is not a square image"
        self.n_rows = self.n_col = input_size//model_patch_size
        self.input_size = (input_size//model_patch_size)**2
        self.model_patch_size = model_patch_size 
        self.masks = self.__generate_mask()

    def __flatten_mask(self, masks): 
        new_mask = []
        # new_mask = [int((mask[0]*n_col)+mask[1]) for mask in masks]
        for mask in masks:
            assert mask[0] < self.n_rows and mask[1] < self.n_col #sanity check  
            flat_idx = int((mask[0]*self.n_col)+mask[1])
            new_mask.append(flat_idx)
        return np.array(new_mask)

    def __cover_outline(self, top_left:int, bottom_right:int): 
        top = [torch.tensor([p, top_left]) for p in range(top_left, bottom_right+1)]
        left = [torch.tensor([top_left, p]) for p in range(top_left, bottom_right+1)]
        right = [torch.tensor([bottom_right, p]) for p in range(top_left, bottom_right+1)]
        bottom = [torch.tensor([p, bottom_right]) for p in range(top_left, bottom_right+1)]
        mask_idx = torch.stack(top+left+right+bottom+ [torch.tensor([top_left,top_left]), torch.tensor([bottom_right, bottom_right])])
        mask_idx = torch.unique(mask_idx, dim=0)
        return mask_idx
     
    def __generate_mask(self): 
        masks = [] 
        # top_left, bottom_right = (self.n_rows//2)-1, (self.n_rows//2)+1
        top_left = bottom_right = (self.n_rows//2)
        masked_idxs_all = self.__cover_outline(top_left, bottom_right)
        for _ in range(self.n_rows//2):
            # masked_idxs = self.__cover_outline(top_left, bottom_right)
            # flatt_idx = self.__flatten_mask(masked_idxs)
            # mask = np.zeros(self.input_size, dtype=int)
            # mask[flatt_idx] = 1
            mask = np.zeros((self.n_rows, self.n_rows), dtype=int)
            mask[masked_idxs_all[:, 0], masked_idxs_all[:, 1]] = 1
            
            masks.append(mask)
            top_left = torch.min(masked_idxs_all)-1
            bottom_right = torch.max(masked_idxs_all)+1 
            masked_idxs_all = torch.unique(
                torch.concat([masked_idxs_all, self.__cover_outline(top_left, bottom_right)], dim=0)
                , dim=0
                )
        return masks
    
    def __call__(self): 
        return torch.tensor(self.masks)
    
class ExpandingMaskTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = ExpandingMaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        img = torch.stack([img for _ in range(len(mask))])
        return img,mask


def em_collate_fn(batch):
    batch = collate_fn(batch)
    GB, MB, C, H, W = batch[0].shape
    batch[0] = batch[0].view(GB*MB, C, H, W)
    batch[1] = batch[1].view(GB*MB, -1)
    return batch

def build_loader_ExpandingMask(config, logger):
    transform = ExpandingMaskTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, 
                            collate_fn=em_collate_fn)
    return dataloader

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader