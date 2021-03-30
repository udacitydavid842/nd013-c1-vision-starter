import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # get data file names from data_dir path
    files = [filename for filename in glob.glob(f'{data_dir}/*.tfrecord')]
    
    # spliting file names
    train_files, val_file, test_file = np.split(files, [int(.75*len(files)), int(.9*len(files))])
    
    # create dirs and move data files into them
    train = os.path.join(data_dir, 'train')
    os.makedirs(train, exist_ok=True)
    for file in train_files:
        shutil.move(file, train)
    
    val = os.path.join(data_dir, 'val')
    os.makedirs(val, exist_ok=True)
    for file in val_file:
        shutil.move(file, val)
    
    test = os.path.join(data_dir, 'test')
    os.makedirs(test, exist_ok=True)
    for file in test_file:
        shutil.move(file, test) 
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
