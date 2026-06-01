from argparse import ArgumentParser
import torch
from exp import exp_DCTF
import os

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = ArgumentParser()

# ========== 数据参数 ==========
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, default='CC-PV.csv')
parser.add_argument('--features', type=str, default='MS')
parser.add_argument('--freq', type=str, default='t')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--cols', type=str, nargs='+')

# ========== 序列参数 ==========
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=24)

# ========== 模型参数（不动） ==========
parser.add_argument('--c_in', type=int, default=8)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--d_ff', type=int, default=1024)
parser.add_argument('--input', type=int, default=8)

# ========== 训练参数 ==========
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=150)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--inverse', action='store_true', default=False)

# ========== GPU ==========
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1')

# ========== 路径 ==========
parser.add_argument('--save_path', type=str, default='./checkpoints/')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# ========== 兼容旧参数 ==========
parser.add_argument('--down_sampling_method', type=str, default='avg')
parser.add_argument('--use_norm', type=int, default=1)
parser.add_argument('--sampling_layers', type=int, default=2)

# ========== 新模型参数 ==========
parser.add_argument('--decomp_kernels', type=int, nargs='+', default=[7, 13, 25, 49])
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--decomp_gate_hidden', type=int, default=32)
parser.add_argument('--model_name', type=str, default='DCTF-Net')

if __name__ == '__main__':
    args = parser.parse_args()

    exp = exp_DCTF.Exp_model(args)

    dataset_name = os.path.splitext(args.data_path)[0]
    settings = f"{dataset_name}_{args.model_name}"

    exp.train(settings)
    exp.test(settings)
