import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

from models.deep_quantreg_models import DeepQuantReg

from utils.metrics import get_ci, get_mse, get_ql, get_ci_cens, get_ci_ipcw, get_censdcal
from utils.helpers import WeightedLoss as WeightedLoss
from utils.helpers import WeightedLoss2 as WeightedLoss2
from utils.helpers import multi_censored_quantile_loss

from efficient_kan import KAN as ekan
from torch.autograd import Variable
import math

import os
import csv
from filelock import FileLock

from datetime import datetime

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

def plot_training_results(n_epoch, train_avg_losses, valid_losses, train_losses, train_mse_trues, valid_mse_trues, train_mse_trues_5, valid_mse_trues_5, train_mses, valid_mses, train_cis, valid_cis, model_type, dataset, dataset_str, dim_feedforward, lr, dropout, n, seed, jpg_name, result_path):
    epochs = range(n_epoch)
    
    plt.rcParams.update(plt.rcParamsDefault)
    if dataset == 'deepquantreg' or dataset == 'simulated':
        plt.figure(figsize=(16, 12)) # width 16, height 12
    else:
        plt.figure(figsize=(16, 8))
    
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_avg_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training avg and Validation Loss')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_mses, label='Train mean MMSE')
    plt.plot(epochs, valid_mses, label='Validation mean MMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training and Validation MMSE')
    plt.legend()

    plt.subplot(3, 2, 4)
    # plt.plot(epochs, train_cis, label='Train CI', linestyle='dashed')
    plt.plot(epochs, train_cis, label='Train mean CI')
    plt.plot(epochs, valid_cis, label='Validation mean CI')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation CI')
    plt.legend()

    if dataset == 'deepquantreg' or dataset == 'simulated':
        plt.subplot(3, 2, 5)
        plt.plot(epochs, train_mse_trues, label='Train True MSE')
        plt.plot(epochs, valid_mse_trues, label='Validation True MSE')
        plt.xlabel('Epoch')
        plt.ylabel('True MSE')
        plt.title('Training and Validation True MSE')
        plt.legend()
        
        plt.subplot(3, 2, 6)
        plt.plot(epochs, train_mse_trues_5, label='Train True MSE 50 Quantile')
        plt.plot(epochs, valid_mse_trues_5, label='Validation True MSE 50 Quantile')
        plt.xlabel('Epoch')
        plt.ylabel('True MSE 50 Quantile')
        plt.title('Training and Validation True MSE 50 Quantile')
        plt.legend()
    
    plt.tight_layout()
    
    # Save the plot with the specified filename format
    filename = f"{result_path}/{jpg_name}.jpg"
    plt.savefig(filename)
    # plt.show()

def deep_quantreg(dataset, dataset_str, train_df, valid_df, test_df, model_type, tau_seq, dim_feedforward=512, nhead=2, n_epoch=100, bsize=64, opt="Adam", uncertainty=False, dropout=0.2, verbose=0, penalty=2, lr=0.001, weight_decay=0, layers=2, n=1500, seed=42, jpg_name='None', result_path='./', acfn='relu', device='cpu',theta=9999,loss_fn='nohuber',grid_size=5,expname='None'):
    
    # def convert_d(pred, tau_seq=tau_seq):
    #     if len(tau_seq) == 1:
    #         return pred.unsqueeze(1)
    #     else:
    #         return pred
    num_workers = 0
    if device=='gpu':
        print('torch.cuda.is_available():', torch.cuda.is_available())
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_workers = 0
    print('Using device:', device)
    # if device == 'cpu':
    #     torch.set_num_threads(4)
    #     num_workers = 2

    X_train = torch.tensor(train_df["X"], dtype=torch.float32)
    Y_train = torch.tensor(train_df["Y"], dtype=torch.float32)
    E_train = torch.tensor(train_df["E"], dtype=torch.float32)
    W_train = torch.tensor(train_df["W"], dtype=torch.float32)
    # shape of Y_train and E_train: torch.Size([n])
    u1 = torch.max(Y_train[E_train == 0]).item() // 1

    X_valid = torch.tensor(valid_df["X"], dtype=torch.float32)
    Y_valid = torch.tensor(valid_df["Y"], dtype=torch.float32)
    E_valid = torch.tensor(valid_df["E"], dtype=torch.float32)
    W_valid = torch.tensor(valid_df["W"], dtype=torch.float32)
    uv = torch.max(Y_valid[E_valid == 0]).item() // 1
        
    X_test = torch.tensor(test_df["X"], dtype=torch.float32)
    Y_test = torch.tensor(test_df["Y"], dtype=torch.float32)
    E_test = torch.tensor(test_df["E"], dtype=torch.float32)
    W_test = torch.tensor(test_df["W"], dtype=torch.float32)
    u2 = torch.max(Y_test[E_test == 0]).item() // 1

    # Multi-diseases
    num_diseases = Y_train.shape[1]
    print('num_diseases:', num_diseases)
        
    train_dataset = TensorDataset(X_train, Y_train, E_train, W_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=0)

    if dataset == 'deepquantreg' or dataset == 'simulated':
        T_train = torch.tensor(train_df["True_Q"], dtype=torch.float32)
        T_valid = torch.tensor(valid_df["True_Q"], dtype=torch.float32)
        T_test = torch.tensor(test_df["True_Q"], dtype=torch.float32)

    if 'KAN' in model_type:
        model = DeepQuantReg(input_dim=X_train.shape[1], output_dim=len(tau_seq), num_diseases=num_diseases, model_type=model_type, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, layers=layers, acfn=acfn, grid_size=grid_size)
    else:
        model = DeepQuantReg(input_dim=X_train.shape[1], output_dim=len(tau_seq), num_diseases=num_diseases, model_type=model_type, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, layers=layers, acfn=acfn)
    model = model.to(device)
    print(model)

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unsupported optimizer')

    # loss_fn = WeightedLoss(torch.tensor(tau_seq).to(device))
    loss_fn_str = loss_fn
    if loss_fn == 'nohuber':    
        loss_fn = WeightedLoss2(tau_seq)  # Initialize with tau sequence
    elif loss_fn == 'huber':
        loss_fn = WeightedLoss(tau_seq)
    elif loss_fn == 'localweighted':
        loss_fn = multi_censored_quantile_loss

    def compute_predicted_y(feature_vec, delta_mat):
        p = delta_mat.shape[0]
        r = delta_mat.shape[1]
        beta_mat = torch.cumsum(delta_mat, dim=1)
        predicted_y = torch.matmul(feature_vec, beta_mat[1:, :]) + beta_mat[0, :]
        return predicted_y

    def compute_predicted_y_modified(model, X):
        model.eval()
        with torch.no_grad():
            feature_vec = model(X)
            delta_mat = torch.cat((model.output_layer.bias.unsqueeze(0), model.output_layer.weight.t()), dim=0)

            p = delta_mat.shape[0]
            r = delta_mat.shape[1]
            beta_mat = torch.cumsum(delta_mat, dim=1)
            
            # Calculate delta_0_vec_clipped
            delta_vec = delta_mat[1:, 1:]
            delta_0_vec = delta_mat[0, 1:]
            delta_minus_vec = torch.clamp(-delta_vec, min=0) # or torch.max(-delta_vec, torch.tensor(0.0))
            delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)
            delta_minus_vec_sum_reshaped = delta_minus_vec_sum.view_as(delta_0_vec)
            clip_value_max = torch.full_like(delta_0_vec, float('Inf'))
            delta_0_vec_clipped = torch.clamp(delta_0_vec, min=delta_minus_vec_sum_reshaped, max=clip_value_max)
            
            # Calculate predicted_y_modified
            beta_mat_first_row = beta_mat[0, 0].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
            beta_mat_modified = torch.cat([beta_mat_first_row, delta_0_vec_clipped.unsqueeze(0)], dim=1)
            beta_cumsum_part = torch.cumsum(beta_mat_modified, dim=1)
            predicted_y_base = torch.matmul(feature_vec, beta_mat[1:, :])  # Shape: (batch_size, r1)
            predicted_y_modified = predicted_y_base + beta_cumsum_part  # Add the first row separately

        return predicted_y_modified

    train_avg_losses = []
    train_losses = []
    valid_losses = []
    train_mse_trues = []
    valid_mse_trues = []
    train_mse_trues_5 = []
    valid_mse_trues_5 = []
    train_mses = []
    valid_mses = []
    train_cis = []
    valid_cis = []

    for epoch in range(n_epoch):
        model.train()
        epoch_losses = []
        for X_batch, Y_batch, E_batch, W_batch in train_loader:
            X_batch, Y_batch, E_batch, W_batch = X_batch.to(device), Y_batch.to(device), E_batch.to(device), W_batch.to(device)
            optimizer.zero_grad()
            feature_vec = model(X_batch)
            if 'huber' in loss_fn_str:
                total_loss = loss_fn(torch.log(Y_batch), feature_vec, W_batch)
            elif loss_fn_str == 'localweighted':
                total_loss = loss_fn(torch.log(Y_batch), feature_vec, X_batch, E_batch, tau_seq)
            # delta_mat = torch.cat((model.output_layer.bias.unsqueeze(0), model.output_layer.weight.t()), dim=0)
            # # delta_mat = delta_mat.to(device) # Already in device since model is in device
            
            # predicted_y = compute_predicted_y(feature_vec, delta_mat)
            
            # delta_vec = delta_mat[1:, 1:]
            # delta_0_vec = delta_mat[0, 1:]
            # delta_minus_vec = torch.clamp(-delta_vec, min=0)
            # delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)
            # delta_minus_vec_sum_reshaped = delta_minus_vec_sum.view_as(delta_0_vec)
            # clip_value_max = torch.full_like(delta_0_vec, float('Inf'))
            # delta_0_vec_clipped = torch.clamp(delta_0_vec, min=delta_minus_vec_sum_reshaped, max=clip_value_max)

            # loss = loss_fn(torch.log(Y_batch), predicted_y, W_batch)  # Pass batch weights here
            # constraint_term = penalty * torch.mean(torch.abs(delta_0_vec - delta_0_vec_clipped))
            # # From the github code, the default penalty for other parameters is 0. So, I didn't include them in the constraint term.
            # total_loss = loss + constraint_term
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())
        
        if verbose and epoch % 1 == 0:
            avg_loss = np.mean(epoch_losses)
            train_avg_losses.append(avg_loss)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            with torch.no_grad():
                X_train = X_train.to(device)
                X_valid = X_valid.to(device)
                X_test = X_test.to(device)     
                
                # train_predictions = compute_predicted_y_modified(model, X_train).squeeze()
                # valid_predictions = compute_predicted_y_modified(model, X_valid).squeeze()
                # test_predictions = compute_predicted_y_modified(model, X_test).squeeze()
                train_predictions = model(X_train)
                valid_predictions = model(X_valid)
                test_predictions = model(X_test)
                
                train_predictions = train_predictions.cpu().numpy()
                valid_predictions = valid_predictions.cpu().numpy()
                test_predictions = test_predictions.cpu().numpy()

            if 'huber' in loss_fn_str:
                train_loss = loss_fn(np.log(Y_train), torch.tensor(train_predictions), W_train)
                valid_loss = loss_fn(np.log(Y_valid), torch.tensor(valid_predictions), W_valid)
            elif loss_fn_str == 'localweighted':
                train_loss = loss_fn(np.log(Y_train), torch.tensor(train_predictions), X_train, E_train, tau_seq)
                valid_loss = loss_fn(np.log(Y_valid), torch.tensor(valid_predictions), X_valid, E_valid, tau_seq)
            train_losses.append(train_loss.item())
            valid_losses.append(valid_loss.item())
            
            Qpred_train = np.exp(train_predictions).reshape(train_df['X'].shape[0],-1)
            Qpred_valid = np.exp(valid_predictions).reshape(valid_df['X'].shape[0],-1)
            
            if dataset == 'deepquantreg' or dataset == 'simulated':
                # True mse all
                # print((Qpred_train-T_train.numpy()).shape) # (1500,11)
                train_mse_true = np.mean(np.square(Qpred_train-T_train.numpy()))
                valid_mse_true = np.mean(np.square(Qpred_valid-T_valid.numpy()))

                # True mse 0.5 quantile
                train_mse_true_5 = np.mean(np.square(Qpred_train-T_train.numpy()), axis=0)[(len(tau_seq)-1)//2]
                valid_mse_true_5 = np.mean(np.square(Qpred_valid-T_valid.numpy()), axis=0)[(len(tau_seq)-1)//2]

                train_mse_trues.append(train_mse_true)
                valid_mse_trues.append(valid_mse_true)
                train_mse_trues_5.append(train_mse_true_5)
                valid_mse_trues_5.append(valid_mse_true_5)
            else:
                train_mse_true = 0
                valid_mse_true = 0
                train_mse_true_5 = 0
                valid_mse_true_5 = 0

            # # MMSE and CI based on 0.5 quantile and observed data
            # train_predictions_5 = train_predictions[:,(len(tau_seq)-1)//2].squeeze()
            # # print('train_predictions_5', train_predictions_5.shape,Y_train.numpy().shape)
            # train_mse = get_mse(np.log(Y_train.numpy()), train_predictions_5, E_train.numpy())
            # train_ci = get_ci(np.log(Y_train.numpy()), train_predictions_5, E_train.numpy())
            # # train_mses.append(train_mse)
            # # train_cis.append(train_ci)
            
            # valid_predictions_5 = valid_predictions[:,(len(tau_seq)-1)//2].squeeze()
            # # print('train_predictions_5', valid_predictions_5.shape,Y_valid.numpy().shape)
            # valid_mse = get_mse(np.log(Y_valid.numpy()), valid_predictions_5, E_valid.numpy())
            # valid_ci = get_ci(np.log(Y_valid.numpy()), valid_predictions_5, E_valid.numpy())
            # # valid_mses.append(valid_mse)
            # # valid_cis.append(valid_ci)

            train_mmse_l = []
            valid_mmse_l = []
            train_ci_l = []
            valid_ci_l = []
            train_ql_l = []
            valid_ql_l = []

            valid_ci_cens_l = []
            valid_ci_ipcw_l = []

            # observed MMSE and CI for each quantile
            for i in range(len(tau_seq)):
                Qpred_train = np.exp(train_predictions[:,:,i])
                # print(Qpred_train.shape)
                # Qpred_train = np.reshape(Qpred_train, len(Y_train))
                # print(Y_train.numpy().shape,Qpred_train.shape)
                
                train_mse = get_mse(np.log(Y_train.numpy()), np.log(Qpred_train), E_train.numpy())
                train_ci = get_ci(Y_train.numpy(), Qpred_train, E_train.numpy())
                # train_ql = get_ql(Y_train.numpy(), Qpred_train, E_train.numpy(), tau, np.max(Y_train[E_train == 0]))
                train_ql = get_ql(np.log(Y_train.numpy()), np.log(Qpred_train), tau_seq[i], np.log(u1), W_train.numpy())
                train_mmse_l.append(train_mse)
                train_ci_l.append(train_ci)
                train_ql_l.append(train_ql)

                Qpred_valid = np.exp(valid_predictions[:,:,i])
                # Qpred_valid = np.reshape(Qpred_valid, len(Y_valid))
                
                valid_mse = get_mse(np.log(Y_valid.numpy()), np.log(Qpred_valid), E_valid.numpy())
                valid_ci = get_ci(Y_valid.numpy(), Qpred_valid, E_valid.numpy())
                # valid_ql = get_ql(Y_valid.numpy(), Qpred_valid, E_valid.numpy(), tau, np.max(Y_train[E_train == 0]))
                valid_ql = get_ql(np.log(Y_valid.numpy()), np.log(Qpred_valid), tau_seq[i], np.log(uv), W_valid.numpy())

                # valid_ci_cens = get_ci_cens(Y_valid.numpy(), Qpred_valid, E_valid.numpy())
                # valid_ci_ipcw = get_ci_ipcw(Y_valid.numpy(), Qpred_valid, E_valid.numpy())

                valid_mmse_l.append(valid_mse)
                valid_ci_l.append(valid_ci)
                valid_ql_l.append(valid_ql)

                # valid_ci_cens_l.append(valid_ci_cens)
                # valid_ci_ipcw_l.append(valid_ci_ipcw)
            
            train_mses.append(np.mean(train_mmse_l))
            train_cis.append(np.mean(train_ci_l))
            valid_mses.append(np.mean(valid_mmse_l))
            valid_cis.append(np.mean(valid_ci_l))
            train_mse = np.mean(train_mmse_l)
            train_ci = np.mean(train_ci_l)
            valid_mse = np.mean(valid_mmse_l)
            valid_ci = np.mean(valid_ci_l)         

            # print true mse for 0.5 quantile and mean mmse, ci for all quantiles
            print(f'Epoch {epoch}, Train; Loss: {train_loss}, True MSE: {train_mse_true}, True MSE 50 quantile: {train_mse_true_5}, MMSE: {train_mse}, CI: {train_ci}')
            print(f'Epoch {epoch}, Valid; Loss: {valid_loss}, True MSE: {valid_mse_true}, True MSE 50 quantile: {valid_mse_true_5}, MMSE: {valid_mse}, CI: {valid_ci}')

    # Plotting
    plot_training_results(n_epoch, train_avg_losses, valid_losses, train_losses, train_mse_trues, valid_mse_trues, train_mse_trues_5, valid_mse_trues_5, train_mses, valid_mses, train_cis, valid_cis, model_type, dataset, dataset_str, dim_feedforward, lr, dropout, n, seed, jpg_name, result_path)

    model.eval()
    with torch.no_grad():
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        X_valid = X_valid.to(device)

        # train_predictions = compute_predicted_y_modified(model, X_train).squeeze()
        # test_predictions = compute_predicted_y_modified(model, X_test).squeeze()
        # valid_predictions = compute_predicted_y_modified(model, X_valid).squeeze()

        train_predictions = model(X_train)
        valid_predictions = model(X_valid)
        test_predictions = model(X_test)

        train_predictions = train_predictions.cpu().numpy()
        test_predictions = test_predictions.cpu().numpy()
        valid_predictions = valid_predictions.cpu().numpy()
    
    # # Specify the filename
    # filename = f"{result_path}/{jpg_name}.csv"

    # # Save the array to a CSV file
    # np.savetxt(filename, np.exp(test_predictions), delimiter=',')
    
    if dataset == 'deepquantreg' or dataset == 'deepquantreg3' or dataset == 'simulated':
        # True mse all and 0.5 quantile
        Qpred_train = np.exp(train_predictions)
        Qpred_test = np.exp(test_predictions)
        Qpred_valid = np.exp(valid_predictions)

        # print((Qpred_train-T_train.numpy()).shape)  # (1500,11)
        train_mse_true = np.mean(np.square(Qpred_train-T_train.numpy()))
        test_mse_true = np.mean(np.square(Qpred_test-T_test.numpy()))
        valid_mse_true = np.mean(np.square(Qpred_valid-T_valid.numpy()))

        # print((np.mean(np.square(Qpred_train-T_train.numpy()), axis=0)).shape)
        train_mse_true_5 = np.mean(np.square(Qpred_train-T_train.numpy()), axis=0)[(len(tau_seq)-1)//2]
        test_mse_true_5 = np.mean(np.square(Qpred_test-T_test.numpy()), axis=0)[(len(tau_seq)-1)//2]
        valid_mse_true_5 = np.mean(np.square(Qpred_valid-T_valid.numpy()), axis=0)[(len(tau_seq)-1)//2]
        
        train_mse_true_q = [np.mean(np.square(Qpred_train-T_train.numpy()), axis=0)[i] for i in range(len(tau_seq))]
        test_mse_true_q = [np.mean(np.square(Qpred_test-T_test.numpy()), axis=0)[i] for i in range(len(tau_seq))]
        valid_mse_true_q = [np.mean(np.square(Qpred_valid-T_valid.numpy()), axis=0)[i] for i in range(len(tau_seq))]

        # mse of log(T)
        train_mse_true_log = np.mean(np.square(train_predictions-np.log(T_train.numpy())))
        test_mse_true_log = np.mean(np.square(test_predictions-np.log(T_test.numpy())))
        valid_mse_true_log = np.mean(np.square(valid_predictions-np.log(T_valid.numpy())))

        train_mse_true_5_log = np.mean(np.square(train_predictions-np.log(T_train.numpy())), axis=0)[(len(tau_seq)-1)//2]
        test_mse_true_5_log = np.mean(np.square(test_predictions-np.log(T_test.numpy())), axis=0)[(len(tau_seq)-1)//2]
        valid_mse_true_5_log = np.mean(np.square(valid_predictions-np.log(T_valid.numpy())), axis=0)[(len(tau_seq)-1)//2]

        train_mse_true_q_log = [np.mean(np.square(train_predictions-np.log(T_train.numpy())), axis=0)[i] for i in range(len(tau_seq))]
        test_mse_true_q_log = [np.mean(np.square(test_predictions-np.log(T_test.numpy())), axis=0)[i] for i in range(len(tau_seq))]
        valid_mse_true_q_log = [np.mean(np.square(valid_predictions-np.log(T_valid.numpy())), axis=0)[i] for i in range(len(tau_seq))]

        # ql
        # u1 = torch.max(T_train.numpy()[:,i][E_train == 0]).item() // 1
        # u2 = torch.max(T_test.numpy()[:,i][E_test == 0]).item() // 1
        train_ql_true = [get_ql(np.log(T_train.numpy()[:,i]), train_predictions[:,i], q, np.log(torch.max(T_train[:,i][E_train == 0]).item() // 1), W_train.numpy()) for i, q in enumerate(tau_seq)]
        test_ql_true = [get_ql(np.log(T_test.numpy()[:,i]), test_predictions[:,i], q, np.log(torch.max(T_test[:,i][E_test == 0]).item() // 1), W_test.numpy()) for i, q in enumerate(tau_seq)]
        valid_ql_true = [get_ql(np.log(T_valid.numpy()[:,i]), valid_predictions[:,i], q, np.log(torch.max(T_valid[:,i][E_valid == 0]).item() // 1), W_valid.numpy()) for i, q in enumerate(tau_seq)]

        train_ci_true_q = [get_ci(T_train.numpy()[:,i], Qpred_train[:,i], E_train.numpy()) for i in range(len(tau_seq))]
        test_ci_true_q = [get_ci(T_test.numpy()[:,i], Qpred_test[:,i], E_test.numpy()) for i in range(len(tau_seq))]
        valid_ci_true_q = [get_ci(T_valid.numpy()[:,i], Qpred_valid[:,i], E_valid.numpy()) for i in range(len(tau_seq))]
        
        # For simulated models, metrics that can be printed out:
        # for each quantile: 1. MMSE, 2. CI, 3. QL, 4. True MSE, (5. True CI?)
        # for all the quantiles: True MSE
        print("True MSE for all quantiles: train", train_mse_true, 'valid', valid_mse_true, 'test', test_mse_true)
        print("True MSE for each quantile: train", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, train_mse_true_q)]))
        print("True MSE for each quantile: test", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, test_mse_true_q)]))
        print("True MSE for each quantile: valid", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, valid_mse_true_q)]))
        print("True log MSE for all quantiles: train", train_mse_true_log, 'valid', valid_mse_true_log, 'test', test_mse_true_log)
        print("True log MSE for each quantile: train", " | ".join([f"Q{q}: {logmse:.4f}" for q, logmse in zip(tau_seq, train_mse_true_q_log)]))
        print("True log MSE for each quantile: test", " | ".join([f"Q{q}: {logmse:.4f}" for q, logmse in zip(tau_seq, test_mse_true_q_log)]))
        print("True log MSE for each quantile: valid", " | ".join([f"Q{q}: {logmse:.4f}" for q, logmse in zip(tau_seq, valid_mse_true_q_log)]))
        print("True CI for all quantiles: train", np.mean(train_ci_true_q), 'valid', np.mean(valid_ci_true_q), 'test', np.mean(test_ci_true_q))
        print("True CI for each quantile: train", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, train_ci_true_q)]))
        print("True CI for each quantile: test", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, test_ci_true_q)]))
        print("True CI for each quantile: valid", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, valid_ci_true_q)]))
        print("True QL for all quantile: train", np.mean(train_ql_true), 'valid', np.mean(valid_ql_true), 'test', np.mean(test_ql_true))
        print("True QL for each quantile: train", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, train_ql_true)]))
        print("True QL for each quantile: test", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, test_ql_true)]))
        print("True QL for each quantile: valid", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, valid_ql_true)]))
    else:
        train_mse_true = 0
        test_mse_true = 0
        valid_mse_true = 0
        train_mse_true_5 = 0
        test_mse_true_5 = 0
        valid_mse_true_5 = 0
        train_mse_true_q = []
        test_mse_true_q = []
        valid_mse_true_q = []
        train_ci_true_q = []
        test_ci_true_q = []
        valid_ci_true_q = []

    train_mmse_l = []
    test_mmse_l = []
    valid_mmse_l = []
    train_ci_l = []
    test_ci_l = []
    valid_ci_l = []
    train_ql_l = []
    test_ql_l = []
    valid_ql_l = []

    test_ci_cens_l = []
    test_ci_ipcw_l = []
    valid_ci_cens_l = []
    valid_ci_ipcw_l = []

    # observed MMSE and CI for each quantile
    for i in range(len(tau_seq)):
        Qpred_train = np.exp(train_predictions[:,:,i])
        # print(Qpred_train.shape)
        # Qpred_train = np.reshape(Qpred_train, len(Y_train))
        # print(Y_train.numpy().shape,Qpred_train.shape)
        
        train_mse = get_mse(np.log(Y_train.numpy()), np.log(Qpred_train), E_train.numpy())
        train_ci = get_ci(Y_train.numpy(), Qpred_train, E_train.numpy())
        # train_ql = get_ql(Y_train.numpy(), Qpred_train, E_train.numpy(), tau, np.max(Y_train[E_train == 0]))
        train_ql = get_ql(np.log(Y_train.numpy()), np.log(Qpred_train), tau_seq[i], np.log(u1), W_train.numpy())
        train_mmse_l.append(train_mse)
        train_ci_l.append(train_ci)
        train_ql_l.append(train_ql)

        Qpred_test = np.exp(test_predictions[:,:,i])
        # Qpred_test = np.reshape(Qpred_test, len(Y_test))
        
        test_mse = get_mse(np.log(Y_test.numpy()), np.log(Qpred_test), E_test.numpy())
        test_ci = get_ci(Y_test.numpy(), Qpred_test, E_test.numpy())
        # test_ql = get_ql(Y_test.numpy(), Qpred_test, E_test.numpy(), tau, np.max(Y_train[E_train == 0]))
        test_ql = get_ql(np.log(Y_test.numpy()), np.log(Qpred_test), tau_seq[i], np.log(u2), W_test.numpy())

        # test_ci_cens = get_ci_cens(Y_test.numpy(), Qpred_test, E_test.numpy())
        # test_ci_ipcw = get_ci_ipcw(Y_test.numpy(), Qpred_test, E_test.numpy())

        test_mmse_l.append(test_mse)
        test_ci_l.append(test_ci)
        test_ql_l.append(test_ql)

        # test_ci_cens_l.append(test_ci_cens)
        # test_ci_ipcw_l.append(test_ci_ipcw)

        Qpred_valid = np.exp(valid_predictions[:,:,i])
        # Qpred_valid = np.reshape(Qpred_valid, len(Y_valid))

        valid_mse = get_mse(np.log(Y_valid.numpy()), np.log(Qpred_valid), E_valid.numpy())
        valid_ci = get_ci(Y_valid.numpy(), Qpred_valid, E_valid.numpy())
        # valid_ql = get_ql(Y_valid.numpy(), Qpred_valid, E_valid.numpy(), tau, np.max(Y_train[E_train == 0]))
        valid_ql = get_ql(np.log(Y_valid.numpy()), np.log(Qpred_valid), tau_seq[i], np.log(uv), W_valid.numpy())

        # valid_ci_cens = get_ci_cens(Y_valid.numpy(), Qpred_valid, E_valid.numpy())
        # valid_ci_ipcw = get_ci_ipcw(Y_valid.numpy(), Qpred_valid, E_valid.numpy())

        valid_mmse_l.append(valid_mse)
        valid_ci_l.append(valid_ci)
        valid_ql_l.append(valid_ql)

        # valid_ci_cens_l.append(valid_ci_cens)
        # valid_ci_ipcw_l.append(valid_ci_ipcw)

    # test_censdcal_data, test_censdcal = get_censdcal(np.exp(test_predictions), Y_test.numpy(), 1-E_test.numpy(), tau_seq, len(tau_seq))

    print("Observed CI censored for all quantile: test", np.mean(test_ci_cens_l))
    print("Observed CI IPCW for all quantile: test", np.mean(test_ci_ipcw_l))
    print("Observed CI censored for all quantile: valid", np.mean(valid_ci_cens_l))
    print("Observed CI IPCW for all quantile: valid", np.mean(valid_ci_ipcw_l))

    print("Observed average MMSE for all quantile: train", np.mean(train_mmse_l), 'valid', np.mean(valid_mmse_l), 'test', np.mean(test_mmse_l))
    print("Observed average CI for all quantile: train", np.mean(train_ci_l), 'valid', np.mean(valid_ci_l), 'test', np.mean(test_ci_l))
    print("Observed average QL for all quantile: train", np.mean(train_ql_l), 'valid', np.mean(valid_ql_l), 'test', np.mean(test_ql_l))
    print("Observed MMSE for each quantile: train", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, train_mmse_l)]))
    print("Observed CI for each quantile: train", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, train_ci_l)]))
    print("Observed QL for each quantile: train", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, train_ql_l)]))
    print("Observed MMSE for each quantile: test", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, test_mmse_l)]))
    print("Observed CI for each quantile: test", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, test_ci_l)]))
    print("Observed QL for each quantile: test", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, test_ql_l)]))
    print("Observed MMSE for each quantile: valid", " | ".join([f"Q{q}: {mse:.4f}" for q, mse in zip(tau_seq, valid_mmse_l)]))
    print("Observed CI for each quantile: valid", " | ".join([f"Q{q}: {ci:.4f}" for q, ci in zip(tau_seq, valid_ci_l)]))
    print("Observed QL for each quantile: valid", " | ".join([f"Q{q}: {ql:.4f}" for q, ql in zip(tau_seq, valid_ql_l)]))
    # print("Observed CensDCal: test", test_censdcal)

    datestr=datetime.now().date().strftime("%Y-%m-%d")
    timestr=datetime.now().time().strftime("%H:%M:%S")
    # header = ['dataset','dataset_str','model_type','MMSE_test','CI_test','QL_test','MMSE_valid','CI_valid','QL_valid','MMSE_train','CI_train','QL_train','n_quantiles','seed','n','lr','dropout','n_epoch','weight_decay','theta','penalty','acfn','d','layers','nhead','loss_fn','grid_size','date','time']
    # data_row = [dataset, dataset_str, model_type, np.mean(test_mmse_l), np.mean(test_ci_l), np.mean(test_ql_l), np.mean(valid_mmse_l), np.mean(valid_ci_l), np.mean(valid_ql_l), np.mean(train_mmse_l), np.mean(train_ci_l), np.mean(train_ql_l), len(tau_seq), seed, n, lr, dropout, n_epoch, weight_decay, theta, penalty, acfn, dim_feedforward, layers, nhead, loss_fn, grid_size, datestr, timestr]
    
    # Add each quantile's MMSE as a separate column
    quantile_mmse_values = [mse for mse in test_mmse_l]
    quantile_ci_values = [ci for ci in test_ci_l]

    # # Add headers for each quantile's MMSE
    # if len(tau_seq) == 1:
    #     quantile_headers = ["MMSE"]
    #     quantile_headers_CI = ["CI"]
    #     header = ['dataset', 'dataset_str', 'model_type', 'MMSE_test', 'CI_test', 'QL_test', 
    #             'MMSE_valid', 'CI_valid', 'QL_valid', 'MMSE_train', 'CI_train', 'QL_train', 
    #             'quantile', 'seed', 'n', 'lr', 'dropout', 'n_epoch', 'weight_decay', 
    #             'theta', 'penalty', 'acfn', 'd', 'layers', 'nhead', 'loss_fn', 'grid_size', 
    #             'date', 'time'] + quantile_headers + quantile_headers_CI
    #     data_row = [dataset, dataset_str, model_type, np.mean(test_mmse_l), np.mean(test_ci_l), np.mean(test_ql_l), 
    #                 np.mean(valid_mmse_l), np.mean(valid_ci_l), np.mean(valid_ql_l), np.mean(train_mmse_l), 
    #                 np.mean(train_ci_l), np.mean(train_ql_l), tau_seq[0], seed, n, lr, dropout, n_epoch, weight_decay, 
    #                 theta, penalty, acfn, dim_feedforward, layers, nhead, loss_fn, grid_size, datestr, timestr] + quantile_mmse_values + quantile_ci_values
    # else:
    #     quantile_headers = [f"MMSE_Q{q}" for q in tau_seq]
    #     quantile_headers_CI = [f"CI_Q{q}" for q in tau_seq]
    #     header = ['dataset', 'dataset_str', 'model_type', 'MMSE_test', 'CI_test', 'QL_test', 
    #             'MMSE_valid', 'CI_valid', 'QL_valid', 'MMSE_train', 'CI_train', 'QL_train', 
    #             'n_quantiles', 'seed', 'n', 'lr', 'dropout', 'n_epoch', 'weight_decay', 
    #             'theta', 'penalty', 'acfn', 'd', 'layers', 'nhead', 'loss_fn', 'grid_size', 
    #             'date', 'time'] + quantile_headers + quantile_headers_CI
    #     data_row = [dataset, dataset_str, model_type, np.mean(test_mmse_l), np.mean(test_ci_l), np.mean(test_ql_l), 
    #                 np.mean(valid_mmse_l), np.mean(valid_ci_l), np.mean(valid_ql_l), np.mean(train_mmse_l), 
    #                 np.mean(train_ci_l), np.mean(train_ql_l), len(tau_seq), seed, n, lr, dropout, n_epoch, weight_decay, 
    #                 theta, penalty, acfn, dim_feedforward, layers, nhead, loss_fn, grid_size, datestr, timestr] + quantile_mmse_values + quantile_ci_values

    


    # # if model_type=='MLP':
    # #     file_path = result_path+'/summary/summary_'+dataset+'_'+dataset_str+'_samplesize'+n+'_model_'+model_type+'_dim'+str(dim_feedforward)+'_layers'+str(layers)+'_lr'+str(lr)+'_dp'+str(dropout)+'_epoch'+str(n_epoch)+'.csv'
    # # elif 'KAN' not in model_type and 'Trans' in model_type:
    # #     file_path = result_path+'/summary/summary_'+dataset+dataset_str+model_type+'_'+str(nhead)+'_'+str(dim_feedforward)+'.csv'
    # # elif 'KAN' in model_type and 'Trans' in model_type:
    # #     file_path = 
    # # elif 'KAN' in model_type:
    # #     file_path = result_path+'/summary/summary_'+dataset+dataset_str+model_type+grid_size+'.csv'
    # # file_exists = os.path.isfile(file_path)
    
    # # Check if the base folder exists, if not, create it
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    #     print("Base folder created:", result_path)
    # else:
    #     print("Base folder already exists:", result_path)

    # # Define the summary folder path
    # summary_path = os.path.join(result_path, 'summary')

    # # Check if the summary folder exists, if not, create it
    # if not os.path.exists(summary_path):
    #     os.makedirs(summary_path)
    #     print("Summary folder created:", summary_path)
    # else:
    #     print("Summary folder already exists:", summary_path)

    # file_path = result_path+'/summary/summary_'+expname+'.csv'
    # file_exists = os.path.isfile(file_path)
    # lock = FileLock(file_path + ".lock")

    # with lock:
    #     with open(file_path, 'a', newline='') as file:
    #         writer = csv.writer(file)

    #         # Write the header only if the file did not exist
    #         if not file_exists:
    #             writer.writerow(header)

    #         # Write the single row of data
    #         writer.writerow(data_row)
    # # with open(file_path, 'a', newline='') as file:
    # #     writer = csv.writer(file)

    # #     # Write the header only if the file did not exist
    # #     if not file_exists:
    # #         writer.writerow(header)

    # #     # Write the single row of data
    # #     writer.writerow(data_row)
    
    # # if dataset == 'custom':
    # #     if dataset_str == 'metabric':
    # #         # Open the CSV file
    # #         file_path = result_path+'/summary/summary_'+expname+'.csv'
    # #         file_exists = os.path.isfile(file_path)
    # #         with open(file_path, 'a', newline='') as file:
    # #             writer = csv.writer(file)

    # #             # Write the header only if the file did not exist
    # #             if not file_exists:
    # #                 writer.writerow(header)

    # #             # Write the single row of data
    # #             writer.writerow(data_row)

    # if uncertainty:
    #     predmean, lowerCI, upperCI = predict_with_uncertainty(model, X_test)
    #     return Output(predmean, lowerCI, upperCI, test_ci, test_mse, test_ql, train_ci, train_mse, train_ql)
    # else:
    #     return Output(Qpred_test, lower=None, upper=None, ci=test_ci, mse=test_mse, ql=test_ql, mse_true=test_mse_true, mse_true_5=test_mse_true_5, train_ci=train_ci, train_mse=train_mse, train_ql=train_ql, train_mse_true=train_mse_true, train_mse_true_5=train_mse_true_5)

