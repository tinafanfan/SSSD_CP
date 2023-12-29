
from imputers.SSSDS4Imputer import SSSDS4Imputer
import json
import argparse
import torch
import torch.nn as nn
import os
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

ckpt_iter = 'max'

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.json',
                    help='JSON file for configuration')
args = parser.parse_args()

with open(args.config) as f:
    data = f.read()
config = json.loads(data)

model_config = config['wavenet_config']
net = SSSDS4Imputer(**model_config)

print(net)


# load checkpoint
ckpt_path = "./results/mujoco/90/"
local_path = "T200_beta00.0001_betaT0.02"


ckpt_path = os.path.join(ckpt_path, local_path)
if ckpt_iter == 'max':
    ckpt_iter = find_max_epoch(ckpt_path)
model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
checkpoint = torch.load(model_path, map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])

