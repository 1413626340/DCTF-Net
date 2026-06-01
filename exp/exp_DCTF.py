from exp.exp_basic import Exp_Basic
from exp.tools import EarlyStopping, adjust_learning_rate, StandardScaler
from exp.metrics import metric
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt


from models.DCTF_model_cc import DCTFmodel


class Exp_model(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = DCTFmodel(args=self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        model_optim = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

        criterion = nn.MSELoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data,
                    batch_x,
                    batch_y,
                    batch_x_mark
                )

                f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, :, f_dim:]
                true = true[:, :, f_dim:]

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item()
                        )
                    )

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )

                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                            speed,
                            left_time
                        )
                    )

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1,
                    time.time() - epoch_time
                )
            )

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data,
                batch_x,
                batch_y,
                batch_x_mark
            )

            f_dim = -1 if self.args.features == 'MS' else 0
            pred = pred[:, :, f_dim:]
            true = true[:, :, f_dim:]

            loss = criterion(
                pred.detach().cpu(),
                true.detach().cpu()
            )

            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data,
                batch_x,
                batch_y,
                batch_x_mark
            )

            f_dim = -1 if self.args.features == 'MS' else 0
            pred = pred[:, :, f_dim:]
            true = true[:, :, f_dim:]

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('test shape:', preds.shape, trues.shape)

        folder_path = './result/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        mean = np.mean(trues)
        rss = np.sum((trues - preds) ** 2)
        tss = np.sum((trues - mean) ** 2)
        r2 = 1 - (rss / tss)

        print(
            'mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, r2:{}'.format(
                mae,
                mse,
                rmse,
                mape,
                mspe,
                r2
            )
        )

        np.save(
            folder_path + 'metrics.npy',
            np.array([mae, mse, rmse, mape, mspe, r2])
        )

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae

    def parament_caluate(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"参数总数: {total_params:,}")

        trainable_params = sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )
        print(f"可训练参数数量: {trainable_params:,}")

    def calculate_model_size(self, model, dtype=torch.float32):
        if dtype == torch.float32:
            bytes_per_param = 4
        elif dtype == torch.float64:
            bytes_per_param = 8
        elif dtype == torch.float16:
            bytes_per_param = 2
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")

        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = total_params * bytes_per_param
        total_mb = total_bytes / (1024 * 1024)

        print(f"模型参数占用内存: {total_mb:.2f} MB")

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        batch_y = batch_y.to(self.device)

        return outputs, batch_y

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }

        Data = data_dict[self.args.data]

        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred

        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader
