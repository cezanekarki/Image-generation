import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import yaml

__C = edict()
cfg = __C

__C.DATASET_NAME = 'coco'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = False
__C.WORKERS = 6

__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = '../data/coco/'
__C.IMAGE_DIR = '../data/train2014'
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 1024
__C.STAGE = 2


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = False
__C.TRAIN.BATCH_SIZE = 10
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 128
__C.GAN.DF_DIM = 96
__C.GAN.GF_DIM = 192
__C.GAN.R_NUM = 2

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024


def _merge_a_into_b(a, b):

    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def conf(filename):

    with open(filename, 'r') as yaml_stream:
        yaml_cfg = edict(yaml.load(yaml_stream,Loader=yaml.SafeLoader))
    
    _merge_a_into_b(yaml_cfg, __C)
    