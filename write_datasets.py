import os
from argparse import ArgumentParser
from typing import List
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

Section('data', 'Data arguments').params(
    dataset=Param(str, 'Which dataset to write', required=True),
    data_dir=Param(str, 'Where to store the downloaded data', default='./tmp_data'),
    write_dir=Param(str, 'Where to store the ffcv files', default='./data')
)

def write_dataset(dataset: str, split: str, data_dir: str, write_dir: str):
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(write_dir, exist_ok=True)
    
    # Set up dataset specific parameters
    is_train = split == 'train'
    
    # Select and load dataset
    if dataset == 'cifar10':
        ds = CIFAR10(
            root=data_dir,
            train=is_train,
            download=True
        )
        write_path = os.path.join(write_dir, f'cifar10_{split}.ffcv')
        
    elif dataset == 'cifar100':
        ds = CIFAR100(
            root=data_dir,
            train=is_train,
            download=True
        )
        write_path = os.path.join(write_dir, f'cifar100_{split}.ffcv')
        
    elif dataset == 'imagenette':
        from torchvision.datasets import ImageFolder
        # For Imagenette, we need to download it separately
        if not os.path.exists(os.path.join(data_dir, 'imagenette2')):
            print("Downloading Imagenette dataset...")
            import urllib.request
            import tarfile
            
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
            filename = os.path.join(data_dir, 'imagenette2.tgz')
            
            # Download the file
            urllib.request.urlretrieve(url, filename)
            
            # Extract the file
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=data_dir)
                
            # Remove the tar file
            os.remove(filename)
        
        # Set up the correct path for train/val
        split_folder = 'train' if is_train else 'val'
        ds = ImageFolder(os.path.join(data_dir, 'imagenette2', split_folder))
        write_path = os.path.join(write_dir, f'imagenette_{split}.ffcv')
    
    print(f'Writing dataset {dataset} ({split}) to {write_path}')
    
    # Write dataset to FFCV format
    writer = DatasetWriter(
        write_path,
        {
            'image': RGBImageField(),
            'label': IntField()
        },
        num_workers=16
    )
    
    writer.from_indexed_dataset(ds)
    print(f'Finished writing dataset to {write_path}')

@param('data.dataset')
@param('data.data_dir')
@param('data.write_dir')
def main(dataset, data_dir, write_dir):
    # Process both train and val splits
    for split in ['train', 'val']:
        write_dataset(dataset, split, data_dir, write_dir)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Write datasets to FFCV format')
    
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    
    args = config.get()
    main(**vars(args.data))