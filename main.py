# coding=utf-8
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gnn import DELADTI
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from tqdm import tqdm
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, f1_score


def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            data = data.to(device)
            labels = data.y.long()

            predicted_scores = model(data)
            loss = Loss(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC

def test_model(dataset_load, save_path, DATASET, LOSS, dataset="Train", lable="best", save=True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model, test_pbar, LOSS)
    # 预测的label
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test

def show_result(DATASET, lable, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.3f}({:.3f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.3f}({:.3f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.3f}({:.3f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.3f}({:.3f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.3f}({:.3f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.3f}({:.3f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.3f}({:.3f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.3f}({:.3f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.3f}({:.3f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.3f}({:.3f})'.format(PRC_mean, PRC_var))

def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]
    return trainset, validset

modeling = [DELADTI]
model_st = modeling[0].__name__
cuda_name = "cuda:0"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.00003
    # NUM_EPOCHS = 1000
    NUM_EPOCHS = 200
    K_Fold = 5

    dataset = "Davis"
    if dataset == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
    elif dataset == "DrugBank":
        weight_CE = None
    elif dataset == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    processed_data_file = 'data/processed/' + dataset + '.pt'
    if (not os.path.isfile(processed_data_file)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset)
        as_data = TestbedDataset(root='data', dataset='AS_case')
        for i_fold in range(K_Fold):
            print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
            # train/dev/test 
            train_dataset, test_dataset = get_kfold_data(i_fold, train_data)
            train_data_len = len(train_dataset)
            valid_size = int(0.2*train_data_len)
            train_size = train_data_len - valid_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
 

            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
            
            """ create model"""
            model = modeling[0]().to(device)
            """weight initialize"""
            weight_p, bias_p = [], []
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for name, p in model.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            weight_decay = 1e-4

            Loss = nn.CrossEntropyLoss(weight=weight_CE)
            optimizer = torch.optim.AdamW([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}],
                                          lr=LR)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=LR * 10,
                                                    cycle_momentum=False, step_size_up=train_size // TRAIN_BATCH_SIZE)

            save_path = "./" + dataset + "/{}".format(i_fold)
            note = ''
            writer = SummaryWriter(log_dir=save_path, comment=note)
            
            """Output files."""
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # file_results = save_path+'The_results_of_whole_dataset.txt'

            early_stopping = EarlyStopping(savepath = save_path, patience=50, verbose=True, delta=0)
            
            for epoch in range(1, NUM_EPOCHS+1):
                trian_pbar = tqdm(
                    enumerate(BackgroundGenerator(train_loader)),total=len(train_loader))
                # train the model
                train_losses_in_epoch = []  
                model.train()
                for batch_idx, data in trian_pbar:
                    data = data.to(device)
                    optimizer.zero_grad()
                    predicted_interaction = model(data)    
                    train_loss = Loss(predicted_interaction, data.y.long())
                    train_losses_in_epoch.append(train_loss.item())
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss  
                writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
            
                # valid the model
                model.eval()
                Y, P, S = [], [], []
                valid_losses_in_epoch = []
                with torch.no_grad():
                    for data in valid_loader:
                        data = data.to(device)
                        valid_labels = data.y.long()
                        valid_scores = model(data)
                        valid_loss = Loss(valid_scores, valid_labels)
                        valid_labels = valid_labels.to('cpu').data.numpy()
                        valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                        valid_predictions = np.argmax(valid_scores, axis=1)
                        valid_scores = valid_scores[:, 1]
                        valid_losses_in_epoch.append(valid_loss.item())
                        Y.extend(valid_labels)
                        P.extend(valid_predictions)
                        S.extend(valid_scores)

                Precision_dev = precision_score(Y, P)
                Reacll_dev = recall_score(Y, P)
                Accuracy_dev = accuracy_score(Y, P)
                AUC_dev = roc_auc_score(Y, S)
                tpr, fpr, _ = precision_recall_curve(Y, S)
                PRC_dev = auc(fpr, tpr)
                valid_loss_a_epoch = np.average(valid_losses_in_epoch)  

                epoch_len = len(str(NUM_EPOCHS))

                print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                            f'train_loss: {train_loss_a_epoch:.5f} ' +
                            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                            f'valid_AUC: {AUC_dev:.5f} ' +
                            f'valid_PRC: {PRC_dev:.5f} ' +
                            f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                            f'valid_Precision: {Precision_dev:.5f} ' +
                            f'valid_Reacll: {Reacll_dev:.5f} ')
                
                writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
                writer.add_scalar('Valid AUC', AUC_dev, epoch)
                writer.add_scalar('Valid AUPR', PRC_dev, epoch)
                writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
                writer.add_scalar('Valid Precision', Precision_dev, epoch)
                writer.add_scalar('Valid Reacll', Reacll_dev, epoch)
                # writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)
                print(print_msg)
                
                early_stopping(valid_loss_a_epoch, model, epoch)

            validset_test_stable_results,_,_,_,_,_ = test_model(valid_loader, save_path, dataset, Loss, dataset="Valid", lable="stable")
            testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
                test_model(test_loader, save_path, dataset, Loss, dataset="Test", lable="stable")

            AUC_List_stable.append(AUC_test)
            Accuracy_List_stable.append(Accuracy_test)
            AUPR_List_stable.append(PRC_test)
            Recall_List_stable.append(Recall_test)
            Precision_List_stable.append(Precision_test)
            with open(save_path + "The_results_of_whole_dataset.txt", 'a') as f:
                f.write("Test the stable model" + '\n')   
                f.write(testset_test_stable_results + '\n')

    show_result(dataset, "stable", Accuracy_List_stable, Precision_List_stable, Recall_List_stable, AUC_List_stable, AUPR_List_stable) 