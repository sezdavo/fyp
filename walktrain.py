# This script handles the training process
import argparse
import math
import time
from tqdm import tqdm
import numpy as np
import random
import os
from os import walk
# import dill as pickle
import pickle as pkl
import wandb
# import cupy as cu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import walktransformer as architecture
from optim import ScheduledOptim
from optim2 import CosineAnnealingWarmupRestarts
#WANDB
wandb.init(project='fyp', entity='sezdavo')

# DEFINE FUNCTIONS
# Function for finding max sequence length in batch for padding
def getLengths(batch):
    # Define length array for determining the maximum sequence length
    arrayLengths = []
    # Loop through batch items and obtain max length
    for item in batch:
        # package = np.load('Users/esd27/piedata/' + item, allow_pickle=True)
        with open('/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk/' + item, "rb") as f:
            package = pkl.load(f)
            array = package[0]
            arrayLengths.append(len(array))
    # Calculate max length
    # maxLength = np.amax(arrayLengths)
    maxLength = max(arrayLengths)

    return maxLength

# Function for unloading required pieces of data from .npy file
def unloadData(item):
    # Load data package from .npy file
    # package = np.load('Users/esd27/piedata/' + item, allow_pickle=True)
    with open('/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk/' + item, "rb") as f:
        package = pkl.load(f)
        array = package[0]
        truth = torch.tensor(package[1]).float()
        length = len(array)
        query = array[length-1]

    return array, query, truth, length

# Function for padding array to max sequence length
def padArray(maxLength, length, array):
    # Calculate number of padding chunks required
    paddingNumber = maxLength - length
    # Define padding array
    # padding = np.array([0, 0, 0, 0, 0, 0])
    padding = [0, 0, 0, 0, 0, 0]
    # If padding required add padding
    if paddingNumber > 0:
        for i in range(paddingNumber):
            # Insert padding
            # array = np.insert(array, 0, padding, axis = 0)
            array.insert(0, padding)
        newArray = array
    else:
        newArray = array
    
    return newArray

# Gets position vectors for positional embedding
def getPositions(seq):
    positions = []
    posCounter = 0
    for s in seq:
        pos = []
        for i in s:
            pos.append(int(round((i[0].item()*30), 1)))
        pos = torch.tensor(pos)
        if posCounter == 0:
            positions = pos
            posCounter += 1
        elif posCounter == 1:
            positions = torch.stack((positions, pos), dim = 0)
            posCounter += 1
        else:
            positions = torch.cat((positions, pos.unsqueeze(0)), dim = 0)
            posCounter += 1
    
    return positions

def buildBatch(batch, device):
    
    # Get maximum sequence length from items in batch
    maxLength = getLengths(batch)
    batchSize = len(batch)
    # Define batch counter for stacking
    count = 0
    for item in batch:
        # Load item and extract the array, the query and the prediction
        array, query, truth, length = unloadData(item)
        # Pad to maximum length
        array = padArray(maxLength, length, array)
        # Create example
        # seqItem = torch.tensor(array).float().to(device)
        # qItem = torch.tensor(query).unsqueeze(0).float().to(device)
        posItem = []
        boxItem = []
        speedItem = []
        for a in array:
            posItem.append([a[1], a[2]])
            boxItem.append([a[3]])
            speedItem.append([a[4]])
        posItem = torch.tensor(posItem).float().to(device)
        boxItem = torch.tensor(boxItem).float().to(device)
        speedItem = torch.tensor(speedItem).float().to(device)
        truthItem = truth.unsqueeze(0).to(device)
        # Stack examples
        if count == 0:
            # seq = seqItem.to(device)
            # q = qItem.to(device)
            pos = posItem.to(device)
            box = boxItem.to(device)
            speed = speedItem.to(device)
            tru = truthItem.to(device)
            count += 1
        elif count == 1:
            # seq = torch.stack((seq, seqItem), dim=0).to(device)
            # q = torch.stack((q, qItem), dim=0).to(device)
            pos = torch.stack((pos, posItem), dim=0).to(device)
            box = torch.stack((box, boxItem), dim=0).to(device)
            speed = torch.stack((speed, speedItem), dim=0).to(device)
            tru = torch.stack((tru, truthItem), dim=0).to(device)
            count += 1
        else:
            # seq = torch.cat((seq, seqItem.unsqueeze(0)), dim=0).to(device)
            # q = torch.cat((q, qItem.unsqueeze(0)), dim=0).to(device)
            pos = torch.cat((pos, posItem.unsqueeze(0)), dim=0).to(device)
            box = torch.cat((box, boxItem.unsqueeze(0)), dim=0).to(device)
            speed = torch.cat((speed, speedItem.unsqueeze(0)), dim=0).to(device)
            tru = torch.cat((tru, truthItem.unsqueeze(0)), dim=0).to(device)
            count += 1
    
    # return seq, q, tru
    return pos, box, speed, tru


# DEFINE A TRAINING EPOCH
# A training epoch is one single loop through all of the data
# We are definfing the epoch here for use in the loop in the train() function
def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''
    #checked = 0
    model.train()
    # define note keeping variables
    total_loss, n_pred_correct, n_pred_total = 0, 0, 0

    desc = '  - (Training)   '
    running_loss = 0
    # Define batch size
    # A batch is the number of training examples passed to the network at once
    random.shuffle(training_data)
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # LOAD DATA
        # seq = batch['seq']
        # q = batch['q']
        pos = batch['pos']
        box = batch['box']
        speed = batch['speed']
        tru = batch['tru']
        #if checked <= 5:
        #    print(seq[0])
        #    print(q[0])
        #    print(tru[0])
        #   checked += 1
        # Build positions vector
        # positions = getPositions(seq).to(device)

        # forward
        optimizer.zero_grad()
        prediction = model(pos, box, speed).squeeze(0)
        # backward and update parameters
        criterion = nn.BCELoss()
        loss = criterion(prediction, tru)
        # running_loss += loss.item()  
        loss.backward()
        # Optimizer 1
        optimizer.step_and_update_lr()
        # wandb.log({"lr": optimizer._optimizer.param_groups[0]['lr']})
        # note keeping
        # First prediction (action)
        if prediction[0] >= 0.5:
            prediction[0] = 1
        elif prediction[0] <= 0.5:
            prediction[0] = 0
        else:
            pass
        # Compare with truths and keep score
        if prediction[0] == tru[0]:
            n_pred_correct += 1
        
        n_pred_total += 1
        
        total_loss += loss.item()

    loss_per_pred = total_loss/n_pred_total
    accuracy = n_pred_correct/n_pred_total
    return loss_per_pred , accuracy

# DEFINE EPOCH EVALUATION 
def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    # Define note keeping variable
    total_loss, n_pred_total, n_pred_correct = 0, 0, 0 
    random.shuffle(validation_data)
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # LOAD DATA
            #seq = batch['seq']
            #q = batch['q']
            pos = batch['pos']
            box = batch['box']
            speed = batch['speed']
            tru = batch['tru']

            # Build positions vector
            # positions = getPositions(seq).to(device)

            # forward
            prediction = model(pos, box, speed).squeeze(0)
            criterion = nn.BCELoss()
            loss = criterion(prediction, tru)

            # note keeping
            # First prediction (action)
            if prediction[0] >= 0.5:
                prediction[0] = 1
            elif prediction[0] <= 0.5:
                prediction[0] = 0
            else:
                pass
            # Compare with truths and keep score
            if prediction[0] == tru[0]:
                n_pred_correct += 1
            
            n_pred_total += 1

            total_loss += loss.item()
    
    loss_per_pred = total_loss/n_pred_total
    accuracy = n_pred_correct/n_pred_total
   
    return loss_per_pred, accuracy

# change inputs depending on OPTIMIZER (add scheduler for 2)
def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''
    wandb.watch(model)
    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        # Optimizer 2
        # scheduler.step()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        # Optimizer 1
        lr = optimizer._optimizer.param_groups[0]['lr']
        # Optimizer 2
        # lr = scheduler.lr
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]
        
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

        # Optimizer 1
        wandb.log({"val_loss": valid_loss, "train_loss":train_loss, "val_acc": valid_accu, "train_acc":train_accu})

        # Optimizer 2
        # wandb.log({"val_loss": valid_loss, "train_loss":train_loss, "val_acc": valid_accu, "train_acc":train_accu, "lr":lr})

def loadData(device):
    batchSize = 32
    
    mypath = '/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk'
    _, _, filenames = next(walk(mypath))

    print("shuffling files...")
    random.shuffle(filenames)
    print("shuffling files...")
    random.shuffle(filenames)
    
    train_iterator = []
    val_iterator = []
    train_files = []
    val_files = []
    trainSplit = 0.8
    switch = int(0.8*len(filenames))
    counter = 0
    batchCounter = 0
    batch = []
    for f in filenames:
        if f.endswith(".p"):
            
            if counter <= switch:
                if batchCounter < batchSize:
                    batch.append(f)
                    batchCounter += 1
                elif batchCounter == batchSize:
                    # Append batch to iterator
                    train_files.append(batch)
                    # Reset all counters and arrays
                    batch = []
                    batch.append(f)
                    batchCounter = 1
            else:
                if batchCounter < batchSize:
                    batch.append(f)
                    batchCounter += 1
                elif batchCounter == batchSize:
                    # Append batch to iterator
                    val_files.append(batch)
                    # Reset all counters and arrays
                    batch = []
                    batch.append(f)
                    batchCounter = 1
            counter += 1
    counter = 0
    for batch in tqdm(train_files):
        # if counter <= 10:
        pos, box, speed, tru = buildBatch(batch, device)
        build = {"pos": pos,
                "box": box,
                "speed": speed,
                "tru": tru
        }
        train_iterator.append(build)
        # counter += 1
    
    counter = 0
    for batch in tqdm(val_files):
        # if counter <= 10:
        pos, box, speed, tru = buildBatch(batch, device)
        build = {"pos": pos,
                "box": box,
                "speed": speed,
                "tru": tru
        }
        val_iterator.append(build)
        # counter += 1

    return train_iterator, val_iterator

def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=8)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=100000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default='PIE/testout')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #========= Loading Dataset =========#

    # if all((opt.train_path, opt.val_path)):
    #     training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    # elif opt.data_pkl:
    #     training_data, validation_data = prepare_dataloaders(opt, device)
    # else:
    #     raise
    training_data, validation_data = loadData(device)

    print(opt)

    transformer = architecture.Trunk().to(device)
   
    # OPTIMIZER 1
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)

    # OPTIMIZER 2
    # optimizer = optim.SGD(transformer.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5) # lr is min lr
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                       first_cycle_steps=500,
    #                                       cycle_mult=1.0,
    #                                       max_lr=0.1,
    #                                       min_lr=0.01,
    #                                       warmup_steps=100,
    #                                       gamma=1.0)
    
    # train(transformer, training_data, validation_data, optimizer, scheduler, device, opt)

if __name__ == '__main__':
    main()
