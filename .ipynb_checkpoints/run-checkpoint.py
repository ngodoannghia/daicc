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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

import scikitplot as skplt
import matplotlib.pyplot as plt

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(2020)

def denoise(df):
    for col in tqdm(df.columns):
        if col not in ['timestamp','block_id']:
            df[col] = np.floor(df[col]*1000) / 1000
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../..', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='size of each batch during training')
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
    parser.add_argument('--nheads', type=int, default=4, help='number of self-attention heads')
    parser.add_argument('--seed', type=int, default=2020, help='seed')
    parser.add_argument('--pos_encode', type=str, default='LSTM', help='method of positional encoding')
    parser.add_argument('--denoise', type=bool, default=True, help='decrease noise')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# device = torch.device("cpu")


#get features
print("Loading data")
# train = pd.read_csv('analysis/pct_rank/train_pct_rank.csv')[['block_id', 'timestamp', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05', 'sensor_07', 'sensor_09']]
# val = pd.read_csv('analysis/pct_rank/val_pct_rank.csv')[['block_id', 'timestamp', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05', 'sensor_07', 'sensor_09']]
train = pd.read_csv('analysis/train.csv')
val = pd.read_csv('analysis/val.csv')
target_train = pd.read_csv('analysis/train_label.csv')['anomalous'].to_numpy()
target_val = pd.read_csv('analysis/val_label.csv')['anomalous'].to_numpy()

# Add feature
# train_suffle = pd.read_csv('analysis/suffle/train_suffle_100.csv')
# target_train_shuffle = pd.read_csv('analysis/suffle/train_label_suffle_100.csv')

# train_new_data = pd.read_csv('analysis/new_data/new_data.csv')
# target_new_data = pd.read_csv('analysis/new_data/df_new_data_label.csv')

# Combine
# train = pd.concat([train, train_new_data], axis=0)
# target_train = pd.concat([target_train,target_new_data], axis=0)

# target_train = target_train['anomalous'].to_numpy()

# target_train = target_train[:, None]
# target_train = np.repeat(target_train, 10, axis=1)

# target_val = target_val[:, None]
# target_val = np.repeat(target_val, 10, axis=1)

# target_test = pd.read_csv('val_labels.csv')['anomalous'].to_numpy()
#exit()
train['timestamp'] = train['timestamp'].apply(lambda x: str(x).split()[-1].split(':')[0])
val['timestamp'] = val['timestamp'].apply(lambda x: str(x).split()[-1].split(':')[0])

train = pd.get_dummies(train, columns=['timestamp'])
val = pd.get_dummies(val, columns=['timestamp'])


if args.denoise:
    train = denoise(train)
    val = denoise(val)

print("Dropping some features")

train.drop(['block_id'], axis=1, inplace=True)
val = val.drop(['block_id'], axis=1)

print("Normalizing")
# RS = RobustScaler()
RS = MinMaxScaler()
train = RS.fit_transform(train)
val = RS.transform(val)

# MM = MinMaxScaler()
# train = MM.fit_transform(train)
# val = MM.transform(val)

print("Reshaping")
train = train.reshape(-1, 10, train.shape[-1])
val = val.reshape(-1, 10, train.shape[-1])

print(train[0])

# np.save('train',train)
# np.save('test',val)

args.nfeatures=train.shape[-1]
#exit()

from sklearn.model_selection import KFold

kf = KFold(n_splits=args.nfolds,random_state=args.seed,shuffle=True)

#Y= [int(group[i][1][0,playerintestingindex]) for i in range(len(group))]
#exit()
# train=group[list(kf.split(group))[args.fold][0]]
# val=group[list(kf.split(group))[args.fold][1]]

#exit()

# train_features=[train[i] for i in list(kf.split(train))[args.fold][0]]
# val_features=[train[i] for i in list(kf.split(train))[args.fold][1]]
# train_targets=[target_train[i] for i in list(kf.split(target_train))[args.fold][0]]
# val_targets=[target_train[i] for i in list(kf.split(target_train))[args.fold][1]]

train_features = train 
val_features = val 
train_targets = target_train
val_targets = target_val

#exit()

print(f"### in total there are {len(train_features)} in train###")
print(f"### in total there are {len(val_features)} in val###")

#exit()

train_dataset = SAKTDataset(train_features,train_targets)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
del train_features

val_dataset = SAKTDataset(val_features,val_targets)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers)
del val_features

#exit()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#initialize model
model = SAKTModel(args.nfeatures, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
                  max_seq=args.max_seq, nlayers=args.nlayers, rnnlayers=args.rnnlayers,
                  dropout=args.dropout,nheads=args.nheads).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99, weight_decay=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#opt_level = 'O1'
#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
# optimizer = Ranger(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()


# opt_level = 'O1'
# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

model=nn.DataParallel(model)
# model.load_state_dict(torch.load("logs/2023-03-09 17:28:19.907766/LSTM_rnn5_transformer5_models/model0.pth"))
#model.load_state_dict(torch.load('models/model1_epoch6.pth'))

from Logger import *
model_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_models'
log_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_logs'
results_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_val_results'

current_time = datetime.now()
f_logs = 'logs/' + str(current_time)
os.mkdir('logs/' + str(current_time))

os.mkdir(f'{f_logs}/{log_dir}')
logger=CSVLogger(['epoch','train_loss','val_loss','val_auc', 'val_auc_2'],f'{f_logs}/{log_dir}/log_fold{args.fold}.csv')
os.mkdir(f'{f_logs}/{model_dir}')
os.mkdir(f'{f_logs}/{results_dir}')

#exit()

val_metric = -1
best_metric = -1
cos_epoch=int(args.epochs*0.75)
#scheduler=lr_AIAYN(optimizer,args.embed_dim,warmup_steps=3000)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(args.epochs-cos_epoch)*len(train_dataloader))
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
steps_per_epoch=len(train_dataloader)
val_steps=len(val_dataloader)

his_val_loss = []
his_val_metric = []
his_val_metric_2 = []

for epoch in range(args.epochs):
    print(f"Epoch: {epoch}")
    model.train()
    train_loss=0
    t=time.time()
    for step,batch in enumerate(train_dataloader):
        #series=batch.to(device)#.float()
        features,targets=batch
        features=features.to(device)
        targets=targets.to(device)
        #exit()

        optimizer.zero_grad()
        output=model(features,None)
        #exit()
        #exit()
        # print("Output: ", output)
        # print("Target: ", targets)
        loss=criterion(output,targets)#*loss_weight_vector
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #scheduler.step()
        # print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
        #                    .format(step+1, steps_per_epoch, train_loss/(step+1), time.time()-t),end='\r',flush=True)
        if epoch > cos_epoch:
            scheduler.step()
        #break
    # scheduler.step()
    print('')
    train_loss/=(step+1)
    print ("Loss train: {:.3f} Time: {:.1f}"
                           .format(train_loss, time.time()-t))
    #exit()
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    for step,batch in enumerate(val_dataloader):
        features,targets=batch
        features=features.to(device)
        targets=targets.to(device)
        with torch.no_grad():
            output=model(features,None)

            loss=criterion(output,targets)
            val_loss+=loss.item()
            #val_metric.append(MCMAE(output.reshape(-1,4),labels.reshape(-1,4),stds[-4:]))
            preds.append(output.cpu())
            truths.append(targets.cpu())
        print ("Validation Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, val_steps, val_loss/(step+1), time.time()-t),end='\r',flush=True)

    preds=torch.cat(preds).numpy()
    truths=torch.cat(truths).numpy()
    # val_metric=(np.abs(truths-preds)*masks).sum()/masks.sum()#*stds['pressure']
    val_metric = roc_auc_score(truths, preds)
    fpr, tpr, th = metrics.roc_curve(truths, preds)
    val_metric_2 = metrics.auc(fpr, tpr)
    
    #exit()
    print('')
    val_loss/=(step+1)
    
    his_val_loss.append(val_loss)
    his_val_metric.append(val_metric)
    his_val_metric_2.append(val_metric_2)

    logger.log([epoch+1,train_loss,val_loss,val_metric, val_metric_2])
    print(f"Val metric: {val_metric}, Val loss: {val_loss}")

    if val_metric > best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(),f'{f_logs}/{model_dir}/model{args.fold}.pth')
        with open(f'{f_logs}/{results_dir}/fold{args.fold}.p','wb+') as f:
            pickle.dump([preds,truths],f)
        
        fpr, tpr, th = metrics.roc_curve(truths, preds)
        val_metric_2 = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=val_metric_2, estimator_name='LSTM')
        display.plot()
        plt.savefig(f'{f_logs}/auc_of_roc.png')
        plt.clf()
    
plt.plot(his_val_loss, his_val_metric)
plt.savefig(f'{f_logs}/his_val_loss_metric.png')
plt.clf()
plt.plot(his_val_loss, his_val_metric_2)
plt.savefig(f'{f_logs}/his_val_loss_metric_2.png')

    
