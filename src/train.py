import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_forecast

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer

import datetime
from torch.utils.tensorboard import SummaryWriter

import random
from math import floor

# from torchmetrics.regression import MeanAbsolutePercentageError


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k,
          writer):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    # local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
    #                                           diffusion_config["beta_0"],
    #                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    # output_directory = os.path.join(output_directory, local_path)
    # if not os.path.isdir(output_directory):
    #     os.makedirs(output_directory)
    #     os.chmod(output_directory, 0o775)
    # print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    # Move the model to the GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        print(output_directory)
        ckpt_iter = find_max_epoch(output_directory)
        print(ckpt_iter)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            print(model_path)
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    
    ### Custom data loading and reshaping ###
    training_data_load = np.load(trainset_config['train_data_path'])

    batch_size = 80
    training_size = training_data_load.shape[0]
    batch_num = floor(training_size/batch_size)
    print(batch_num)

    index = random.sample(range(training_size), batch_num*batch_size)
    training_data = training_data_load[index,] # 捨棄 training_size 除以 batch_size 的餘數
    training_data = np.split(training_data, batch_num, 0) # split into batch_num batches
    training_data = np.array(training_data)
    training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')
    
    
    
    # training
    if ckpt_iter == -1:
        n_iter = ckpt_iter + 2
    else:
        n_iter = ckpt_iter + 1

    print(f'start the {n_iter} iteration')
    while n_iter < n_iters + 1:
        # loss_all = 0

        for batch in training_data:

            ############ shuffle batch after each epoch ############
            if n_iter % batch_num == 0:
                # print(f'update training batch at {n_iter}')
                index = random.sample(range(training_size), batch_num*batch_size)
                training_data = training_data_load[index,] # 捨棄 training_size 除以 batch_size 的餘數
                training_data = np.split(training_data, batch_num, 0) # split into batch_num batches
                training_data = np.array(training_data)
                training_data = torch.from_numpy(training_data).float().cuda()
            ################################################
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)
            elif masking == 'forecast':
                transposed_mask = get_mask_forecast(batch[0], missing_k)

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)
            # loss = training_loss(net, MeanAbsolutePercentageError().cuda(), X, diffusion_hyperparams,
            #                      only_generate_missing=only_generate_missing)                                 

            loss.backward()
            optimizer.step()

            # loss_all += loss.item()
            writer.add_scalar('Train/Loss', loss.item(), n_iter)

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)
                current_time = datetime.datetime.now()
                print("當前時間:", current_time.strftime("%Y-%m-%d %H:%M:%S"))  

            n_iter += 1

        # loss_mean = loss_all/(training_size/batch_size) # calculate mse of all training data
        # iter_all_dataset = n_iter-1
        # print("iteration: {} \tmean_loss: {}".format(iter_all_dataset, loss_mean))
        # writer.add_scalar('Train/Loss', loss_mean, iter_all_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
                        help='JSON file for configuration')

    # parser.add_argument('--logdir', default='log')
    args = parser.parse_args()

    # writer = SummaryWriter(args.logdir)
    

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)


    # 建立 output directionary
    local_path = "T{}_beta0{}_betaT{}".format(config['diffusion_config']["T"],
                                              config['diffusion_config']["beta_0"],
                                              config['diffusion_config']["beta_T"])
    output_directory = os.path.join(config['train_config']['output_directory'], local_path)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # 更新 config file 的值
    config['train_config']['output_directory'] = output_directory

    # 設定 toesorboard 要存在哪
    writer = SummaryWriter(f"{output_directory}/log")
    config['train_config']['writer'] = writer
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']
        
    current_time = datetime.datetime.now()
    print("當前時間:", current_time.strftime("%Y-%m-%d %H:%M:%S"))    

    train(**train_config)

    current_time = datetime.datetime.now()
    print("當前時間:", current_time.strftime("%Y-%m-%d %H:%M:%S"))