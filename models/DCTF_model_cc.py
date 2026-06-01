import torch
import torch.nn as nn
import numpy as np

class AdaptiveMovingAverage(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        padded_x = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded_x.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        residual = x - trend
        return (residual, trend)

class AdaptiveMultiScaleTemporalDecoupling(nn.Module):

    def __init__(self, kernel_sizes=[7, 13, 25, 49]):
        super().__init__()
        self.decomposition_scales = nn.ModuleList([AdaptiveMovingAverage(k) for k in kernel_sizes])
        self.scale_weight = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, target_sequence):
        trend_candidates = []
        for decomposition_scale in self.decomposition_scales:
            _, trend = decomposition_scale(target_sequence)
            trend_candidates.append(trend)
        weights = torch.softmax(self.scale_weight, dim=0)
        trend_component = sum((w * trend for w, trend in zip(weights, trend_candidates)))
        residual_component = target_sequence - trend_component
        return (residual_component, trend_component)

class TrendAnchoringBranch(nn.Module):

    def __init__(self, seq_len, pred_len, dropout=0.05):
        super().__init__()
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trend_component):
        x = trend_component.squeeze(-1)
        x = self.dropout(x)
        trend_prediction = self.temporal_projection(x)
        return trend_prediction.unsqueeze(-1)

class DisturbanceCorrectionBranch(nn.Module):

    def __init__(self, c_in, hidden_dim, pred_len, num_layers=2, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = 4
        self.head_dim = hidden_dim // self.n_heads
        self.disturbance_input_projection = nn.Sequential(nn.Linear(c_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_disturbance_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.disturbance_attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.head_dim), nn.Tanh(), nn.Linear(self.head_dim, 1)) for _ in range(self.n_heads)])
        self.context_redistribution = nn.Linear(hidden_dim * self.n_heads, hidden_dim)
        self.disturbance_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, pred_len))

    def forward(self, residual_covariate_context):
        x = self.disturbance_input_projection(residual_covariate_context)
        rnn_out, h_n = self.temporal_disturbance_encoder(x)
        contexts = []
        for attention_head in self.disturbance_attention_heads:
            scores = attention_head(rnn_out)
            weights = torch.softmax(scores, dim=1)
            context = (rnn_out * weights).sum(dim=1)
            contexts.append(context)
        multi_context = torch.cat(contexts, dim=-1)
        disturbance_context = self.context_redistribution(multi_context)
        disturbance_context = disturbance_context + h_n[-1]
        disturbance_prediction = self.disturbance_decoder(disturbance_context)
        return disturbance_prediction.unsqueeze(-1)

class FrequencyDomainCompensationBranch(nn.Module):

    def __init__(self, c_in, hidden_dim, seq_len, pred_len, modes=32, freq_drop=0.15, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_freqs = seq_len // 2 + 1
        self.freq_drop = freq_drop
        self.modes = modes
        self.frequency_input_projection = nn.Linear(c_in, hidden_dim)
        self.frequency_complex_weight = nn.Parameter(torch.randn(self.max_freqs, hidden_dim, dtype=torch.cfloat) * 0.02)
        self.temporal_horizon_projection = nn.Linear(seq_len, pred_len)
        self.frequency_decoder = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, residual_covariate_context):
        B, L, _ = residual_covariate_context.shape
        x = self.frequency_input_projection(residual_covariate_context)
        x_ft = torch.fft.rfft(x, dim=1)
        compensated_ft = x_ft * self.frequency_complex_weight
        if self.training and self.freq_drop > 0:
            mask = torch.bernoulli(torch.ones(1, self.max_freqs, 1, device=x.device) * (1.0 - self.freq_drop))
            mask[:, 0, :] = 1.0
            compensated_ft = compensated_ft * mask
        compensated_sequence = torch.fft.irfft(compensated_ft, n=L, dim=1)
        compensated_sequence = self.temporal_horizon_projection(compensated_sequence.permute(0, 2, 1)).permute(0, 2, 1)
        frequency_prediction = self.frequency_decoder(compensated_sequence)
        return frequency_prediction

class DCTFmodel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.c_in = args.c_in
        self.c_out = args.c_out
        hidden_dim = getattr(args, 'd_model', 64)
        gru_layers = getattr(args, 'rnn_layers', 2)
        dropout = getattr(args, 'dropout', 0.15)
        decomp_kernels = getattr(args, 'decomp_kernels', [7, 13, 25, 49])
        modes = getattr(args, 'modes', 32)
        self.adaptive_temporal_decoupling = AdaptiveMultiScaleTemporalDecoupling(kernel_sizes=decomp_kernels)
        self.trend_anchoring = TrendAnchoringBranch(seq_len=self.seq_len, pred_len=self.pred_len, dropout=dropout * 0.5)
        self.disturbance_correction = DisturbanceCorrectionBranch(c_in=self.c_in, hidden_dim=hidden_dim, pred_len=self.pred_len, num_layers=gru_layers, dropout=dropout)
        self.frequency_domain_compensation = FrequencyDomainCompensationBranch(c_in=self.c_in, hidden_dim=hidden_dim, seq_len=self.seq_len, pred_len=self.pred_len, modes=modes, dropout=dropout)
        if self.c_out > 1:
            self.channel_projection = nn.Linear(1, self.c_out)
        else:
            self.channel_projection = nn.Identity()
        self._print_lightweight_profile(hidden_dim=hidden_dim, gru_layers=gru_layers, modes=modes)

    def _print_lightweight_profile(self, hidden_dim, gru_layers, modes):
        total_params = sum((p.numel() for p in self.parameters()))
        param_mem_mb = total_params * 4 / 1024 ** 2
        macs_trend = self.seq_len * self.pred_len
        macs_disturbance = self.seq_len * self.c_in * hidden_dim + self.seq_len * hidden_dim ** 2 * 3 * gru_layers + hidden_dim * self.pred_len
        macs_fft = self.seq_len * np.log2(self.seq_len if self.seq_len > 0 else 1) * hidden_dim
        macs_frequency = macs_fft * 2 + self.seq_len * hidden_dim + hidden_dim * self.seq_len * self.pred_len
        total_flops = (macs_trend + macs_disturbance + macs_frequency) * 2
        param_efficiency = total_flops / total_params if total_params > 0 else 0
        print(f'\n[DCTF-Net] 模型构建完成 ✅')
        print('  Heterogeneous temporal complexity redistribution enabled')
        print(f'  输入长度: {self.seq_len} | 预测长度: {self.pred_len} | 特征维度: {self.c_in}')
        print('-' * 45)
        print('--- Lightweight Deployment Profile ---')
        print(f'  总参数量   : {total_params / 1000:.2f} K ({total_params:,})')
        print(f'  参数内存   : {param_mem_mb:.4f} MB')
        print(f'  推理 FLOPs : ~{total_flops / 1000000.0:.2f} M (Batch=1)')
        print(f'  参数效率   : {param_efficiency:.1f} FLOPs/Param')
        print('-' * 45 + '\n')

    def instance_normalization(self, x):
        means = x.mean(1, keepdim=True).detach()
        normalized_x = x - means
        stdev = torch.sqrt(torch.var(normalized_x, dim=1, keepdim=True, unbiased=False) + 1e-05)
        normalized_x = normalized_x / stdev
        return (normalized_x, means, stdev)

    def inverse_instance_normalization(self, prediction, means, stdev):
        target_stdev = stdev[:, 0, -1:].unsqueeze(1)
        target_mean = means[:, 0, -1:].unsqueeze(1)
        prediction = prediction * target_stdev.repeat(1, self.pred_len, self.c_out)
        prediction = prediction + target_mean.repeat(1, self.pred_len, self.c_out)
        return prediction

    def temporal_component_decoupling(self, normalized_input):
        X_target = normalized_input[:, :, -1:]
        X_cov = normalized_input[:, :, :-1]
        X_res, X_trend = self.adaptive_temporal_decoupling(X_target)
        return (X_target, X_cov, X_res, X_trend)

    def heterogeneous_temporal_frequency_modeling(self, X_cov, X_res, X_trend):
        residual_covariate_context = torch.cat([X_cov, X_res], dim=-1)
        Z_trend = self.trend_anchoring(X_trend)
        Z_res = self.disturbance_correction(residual_covariate_context)
        Z_freq = self.frequency_domain_compensation(residual_covariate_context)
        return (Z_trend, Z_res, Z_freq)

    def collaborative_forecasting_reconstruction(self, Z_trend, Z_res, Z_freq):
        prediction = Z_trend + Z_res + Z_freq
        prediction = self.channel_projection(prediction)
        return prediction

    def forward(self, orx, orx_mark=None, return_components=False):
        assert orx.shape[-1] == self.c_in, f'输入张量特征数({orx.shape[-1]})与设定(c_in={self.c_in})不符!'
        normalized_input, means, stdev = self.instance_normalization(orx)
        X_target, X_cov, X_res, X_trend = self.temporal_component_decoupling(normalized_input)
        Z_trend, Z_res, Z_freq = self.heterogeneous_temporal_frequency_modeling(X_cov, X_res, X_trend)
        dec_out = self.collaborative_forecasting_reconstruction(Z_trend, Z_res, Z_freq)
        dec_out = self.inverse_instance_normalization(dec_out, means, stdev)
        if return_components:
            return (dec_out, Z_trend, Z_res, Z_freq)
        return dec_out
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--c_in', type=int, default=11)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--decomp_kernels', type=int, nargs='+', default=[7, 13, 25, 49])
    args = parser.parse_args([])
    print('=' * 70)
    model = DCTFmodel(args)
    B = 32
    x = torch.randn(B, 96, 11)
    model.eval()
    out_eval = model(x)
    print(f'\n[Eval Mode] 输出形状: {out_eval.shape}')
    model.train()
    out_train = model(x)
    print(f'[Train Mode] 输出形状: {out_train.shape}')
    assert out_train.shape == (B, 24, 1), '输出形状错误'
    print('✅ DCTF-Net 测试全部通过！')
