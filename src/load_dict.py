from imputers.SSSDS4Imputer import SSSDS4Imputer
import json
import argparse
import torch
import torch.nn as nn
import os
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.json',
                    help='JSON file for configuration')
parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                    help='Which checkpoint to use; assign a number or "max"')
# parser.add_argument('-n', '--num_samples', type=int, default=500,
#                     help='Number of utterances to be generated')
args = parser.parse_args()

# Parse configs. Globals nicer in this case
with open(args.config) as f:
    data = f.read()
config = json.loads(data)
print(config)

global model_config
model_config = config['wavenet_config']
net = SSSDS4Imputer(**model_config).cuda()

# load checkpoint
local_path = "T200_beta00.0001_betaT0.02"
ckpt_path="./results/ettm1/"
ckpt_iter = "10000"
ckpt_path = os.path.join(ckpt_path, local_path)
model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))

checkpoint = torch.load(model_path, map_location='cpu')
print("checkpoint")
net.load_state_dict(checkpoint['model_state_dict'])
print('Successfully loaded model at iteration {}'.format(ckpt_iter))