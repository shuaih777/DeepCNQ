import argparse
import json
import os
import torch
import numpy as np

from data.data_loader import data_synthtarget_synthcen, organize_data, load_custom_dataset, load_multi_diseases_dataset
from exp.train import deep_quantreg
from utils.helpers import WeightedLoss, WeightedLoss2


def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")

    # 配置文件参数
    parser.add_argument('--config', type=str, help="Path to the JSON config file")
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, choices=['simulated', 'deepquantreg', 'deepquantreg2', 'custom', 'multi_diseases'], default='multi_diseases',
                        help="Dataset to use. Options: ['simulated','deepquantreg','deepquantreg2','custom']")
    parser.add_argument('--dataset_str', type=str, default='metabric', help="Dataset string identifier")
    
    # 模型参数
    parser.add_argument('--n_quantiles', type=int, default=5, help="Number of quantiles")
    parser.add_argument('--quantiles', type=int, default=50, help='Quantile to be estimated if n_q is 1')
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--n', type=int, default=1500, help="Sample size")
    parser.add_argument('--theta', type=float, default=10, help="Censoring time")
    parser.add_argument('--penalty', type=int, default=5, help="penalty")
    parser.add_argument('--model_type', type=str, default='TransformerPS_gaps', help="Type of model")
    parser.add_argument('--acfn', type=str, default='relu', help="activation function")
    parser.add_argument('--d', type=int, default=100, help="Dimension feedforward")
    parser.add_argument('--layers', type=int, default=2, help="Layers of MLP")
    parser.add_argument('--nhead', type=int, default=4, help="Number of heads in multihead attention")
    parser.add_argument('--grid_size', type=int, default=5, help="Grid size for KAN")
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--n_epoch', type=int, default=2, help="Number of epochs")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity level")
    parser.add_argument('--loss_fn', type=str, default='nohuber', help="Loss function")
    
    # 实验设置
    parser.add_argument('--jpg_name', type=str, default='None', help="Name of figure plot")
    parser.add_argument('--result_path', type=str, default='./results/', help="path to save the results")
    parser.add_argument('--expname', type=str, default='None', help="Experiment name")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of splits for cross-validation")
    parser.add_argument('--n_repeats', type=int, default=1, help="Number of repeats for cross-validation")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument('--early_stopping', type=int, default=10, help="Early stopping patience")
    
    # 保存设置
    parser.add_argument('--save_model', type=bool, default=True, help="Save the model")
    parser.add_argument('--save_results', type=bool, default=True, help="Save the results")
    parser.add_argument('--save_path', type=str, default='results/', help="Path to save results")
    parser.add_argument('--save_model_path', type=str, default='models/', help="Path to save models")
    parser.add_argument('--save_results_path', type=str, default='results/', help="Path to save results")
    parser.add_argument('--save_fig_path', type=str, default='figures/', help="Path to save figures")
    parser.add_argument('--save_fig', type=bool, default=True, help="Save figures")
    parser.add_argument('--save_fig_format', type=str, default='pdf', help="Figure save format")
    parser.add_argument('--save_fig_dpi', type=int, default=300, help="Figure DPI")
    
    args = parser.parse_args()

    # 从JSON配置文件加载参数（如果提供）
    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
        # 使用配置文件中的值更新默认参数
        for key, value in config_args.items():
            setattr(args, key, value)

    # 设置分位数
    if args.n_quantiles == 5:
        args.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    elif args.n_quantiles == 11:
        args.quantiles = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    elif args.n_quantiles == 19:
        args.quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    elif args.n_quantiles == 1:
        args.quantiles = [args.quantiles/100]
    
    return args


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 打印参数配置
    print("Parsed Arguments:")
    print(f"dataset: {args.dataset}")
    print(f"nhead: {args.nhead}")
    print(f"dropout: {args.dropout}")
    print(f"n_quantiles: {args.n_quantiles}")
    print(f"batch_size: {args.batch_size}")
    print(f"n_epoch: {args.n_epoch}")
    print(f"grid_size: {args.grid_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"verbose: {args.verbose}")
    print(f"loss_fn: {args.loss_fn}")
    print(f"expname: {args.expname}")
    print(f"d: {args.d}")
    print(f"dataset_str: {args.dataset_str}")
    print(f"model_type: {args.model_type}")
    print(f"quantiles: {args.quantiles}")
    print(f"n: {args.n}")
    print(f"theta: {args.theta}")
    print(f"layers: {args.layers}")
    print(f"jpg_name: {args.jpg_name}")
    print(f'result_path: {args.result_path}')
    print(f"acfn: {args.acfn}")
    print(f"penalty: {args.penalty}")

    # 确保结果目录存在
    os.makedirs(args.result_path, exist_ok=True)
    
    # 根据数据集类型加载不同的数据
    if args.dataset == 'simulated':
        # 加载模拟数据
        train_df, valid_df, test_df, train_true_q_df, valid_true_q_df, test_true_q_df = data_synthtarget_synthcen(
            n_data=3*args.n, 
            dataset_str=args.dataset_str, 
            x_range=[0,2], 
            seed=args.seed, 
            n_quantiles=args.n_quantiles
        )
        
        # 组织数据
        train_df = organize_data(train_df, time=["y"], event=["obs_indicator"], trt=None)
        train_df['True_Q'] = train_true_q_df.to_numpy()

        valid_df = organize_data(valid_df, time=["y"], event=["obs_indicator"], trt=None)
        valid_df['True_Q'] = valid_true_q_df.to_numpy()

        test_df = organize_data(test_df, time=["y"], event=["obs_indicator"], trt=None)
        test_df['True_Q'] = test_true_q_df.to_numpy()
        
    elif args.dataset == 'custom':
        # 加载自定义数据集
        train_df, valid_df, test_df = load_custom_dataset(args.dataset_str, args.seed)
        
    elif args.dataset == 'multi_diseases':
        # 加载多疾病数据集
        train_df, valid_df, test_df = load_multi_diseases_dataset(args.dataset_str, args.seed)
    
    # 开始训练
    result = deep_quantreg(
        dataset=args.dataset,
        dataset_str=args.dataset_str,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        model_type=args.model_type,
        n_epoch=args.n_epoch,
        bsize=args.batch_size,
        tau_seq=args.quantiles,
        dim_feedforward=args.d,
        uncertainty=False,
        verbose=args.verbose,
        lr=args.lr,
        dropout=args.dropout,
        layers=args.layers,
        n=args.n,
        seed=args.seed,
        jpg_name=args.jpg_name,
        result_path=args.result_path,
        acfn=args.acfn,
        device=args.device,
        nhead=args.nhead,
        grid_size=args.grid_size,
        weight_decay=args.weight_decay,
        theta=args.theta,
        loss_fn=args.loss_fn,
        penalty=args.penalty,
        expname=args.expname
    )
    
    # 打印训练结果
    if args.dataset == 'simulated':
        print(f"{args.model_type} Results:")
        print(f"MMSE: {result.mse:.4f}")
        print(f"CI: {result.ci:.4f}")
        print(f"True MSE: {result.mse_true:.4f}")
        print(f"True MSE 50 quantile: {result.mse_true_5:.4f}")


if __name__ == "__main__":
    main()