import numpy as np
import pandas as pd
import gc
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import time
from Dataset import *
from Network import *
from Functions import *

# from ranger import Ranger
import pickle
import argparse
from sklearn.preprocessing import RobustScaler, normalize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../..', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight dacay used in optimizer')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_seq', type=int, default=64, help='max_seq')
    parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension size')
    #parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
    parser.add_argument('--nlayers', type=int, default=5, help='nlayers')
    parser.add_argument('--rnnlayers', type=int, default=5, help='number of reisdual rnn blocks')
    parser.add_argument('--nfeatures', type=int, default=5, help='amount of features')
    parser.add_argument('--nheads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--seed', type=int, default=2020, help='seed')
    parser.add_argument('--pos_encode', type=str, default='LSTM', help='method of positional encoding')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# device = torch.device("cpu")

test = pd.read_csv('analysis/test.csv')
train = pd.read_csv('analysis/train.csv')

label = pd.read_csv('analysis/train_label.csv')

submission=pd.read_csv('data/sample_submission.csv')

#exit()

print("Dropping some features")

train.drop(['block_id', 'timestamp'], axis=1, inplace=True)
test = test.drop(['block_id', 'timestamp'], axis=1)

print("Normalizing")
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

print("Reshaping")
test = test.reshape(-1, 10, train.shape[-1])
train = train.reshape(-1, 10, train.shape[-1])
args.nfeatures=train.shape[-1]

test_dataset = TestDataset(test)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


model_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_models'
log_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_logs'
results_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_val_results'

# MODELS=[]
# for fold in range(args.nfolds):
#     model = SAKTModel(args.nfeatures, 10, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
#                       max_seq=args.max_seq, nlayers=args.nlayers,rnnlayers=args.rnnlayers,
#                       dropout=args.dropout,nheads=args.nheads).to(device)
#     model=nn.DataParallel(model)
#     model.load_state_dict(torch.load(f'{model_dir}/model{fold}.pth'))
#     model.eval()
#     MODELS.append(model)
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print('Total number of paramters: {}'.format(pytorch_total_params))

#exit()


#MODELS=[]
# for fold in range(args.nfolds):
#     model = SAKTModel(args.nfeatures, 10, 1, embed_dim=args.embed_dim, pos_encode='GRU',
#                       max_seq=args.max_seq, nlayers=args.nlayers,
#                       dropout=args.dropout,nheads=args.nheads).to(device)
#     model=nn.DataParallel(model)
#     model.load_state_dict(torch.load(f'GRU_models/model{fold}.pth'))
#     model.eval()
#     MODELS.append(model)
model = SAKTModel(args.nfeatures, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
                  max_seq=args.max_seq, nlayers=args.nlayers, rnnlayers=args.rnnlayers,
                  dropout=args.dropout,nheads=args.nheads).to(device)
model=nn.DataParallel(model)
model.load_state_dict(torch.load('logs/2023-03-09 17:47:59.196557/LSTM_rnn5_transformer5_models/model0.pth'))
model.eval()
preds=[]
for batch in tqdm(test_dataloader):
    features=batch.to(device)
    #features=features
    with torch.no_grad():
        temp=[]
        # for model in MODELS:
        #     output=model(features,None)
        #     temp.append(output)
        output = model(features, None)
        #temp=torch.mean(torch.stack(temp,0),0)#[0]
        #temp=torch.median(torch.stack(temp,0),0)[0]
        # temp=torch.stack(temp,1)
        #temp=torch.round( (temp - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN

        preds.append(output.cpu())

preds=torch.cat(preds)#.reshape(-1).numpy()

# post_processed=torch.median(preds,1)[0].reshape(-1)
# post_processed=torch.round( (post_processed - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN


# submission = pd.DataFrame(data=preds.numpy(), columns=['anomaly_score'])
# submission['block_id'] = label['block_id'].values
# submission['anomaly'] = label['anomalous'].values
submission['anomaly_score'] = preds.numpy()
submission.to_csv('submission.csv',index=False)

# torch.save(preds,'predictions.b')

# for i in range(args.nfolds):
#     post_processed=preds[:,i].reshape(-1)
#     #post_processed=torch.round( (post_processed - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
#     submission['pressure']=post_processed
#     submission.to_csv(f'submission_fold{i}.csv',index=False)
