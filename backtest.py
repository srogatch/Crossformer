import argparse
import math
import multiprocessing
import os
import pickle
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import torch
from tqdm import tqdm

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import load_args, string_split, StandardScaler

C_SPREAD = 1.5259520851045277178296601486713e-4
MIN_GROWTH_PERC = 0.99
LEVERAGE = 1
BARS_PER_DAY = 1440
COLUMN_NAMES = ['open', 'high', 'low', 'close', 'tick_volume']
BATCH_SIZE = 128
N_BATCHES_AHEAD = 8


def calc_pl(open_price: float, close_price: float, direction: int):
    if direction > 0:
        cur_pl = close_price / ((1 + C_SPREAD) * open_price) - 1
    else:
        cur_pl = open_price / ((1 + C_SPREAD) * close_price) - 1
    return cur_pl


parser = argparse.ArgumentParser(description='CrossFormer Strategy Backtester')
parser.add_argument('--checkpoint_root', type=str, default='./checkpoints', help='location of the trained model')
parser.add_argument(
    '--setting_name', type=str,
    default='Crossformer_prepared-US500_M1_il1440_ol480_sl6_win2_fa10_dm1024_nh16_el6_itr0',
    help='name of the experiment')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--different_split', action='store_true', help='use data split different from training process',
                    default=False)
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='data split of train, vali, test')

parser.add_argument('--inverse', action='store_true', help='inverse output data into the original scale',
                    default=False)
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS',
                    default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False

args.checkpoint_dir = os.path.join(args.checkpoint_root, args.setting_name)
hyper_parameters = load_args(os.path.join(args.checkpoint_dir, 'args.json'))

# load the pre-trained model
args.data_dim = hyper_parameters['data_dim']
args.in_len = hyper_parameters['in_len']
args.out_len = hyper_parameters['out_len']
args.seg_len = hyper_parameters['seg_len']
args.win_size = hyper_parameters['win_size']
args.factor = hyper_parameters['factor']
args.d_model = hyper_parameters['d_model']
args.d_ff = hyper_parameters['d_ff']
args.n_heads = hyper_parameters['n_heads']
args.e_layers = hyper_parameters['e_layers']
args.dropout = hyper_parameters['dropout']
args.baseline = hyper_parameters['baseline']

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    sys.exit(1)

start_date = datetime(2022, 4, 8, 1, 37)
# start_date = datetime(2022, 10, 9)
rates = mt5.copy_rates_range('US500.pro', mt5.TIMEFRAME_M1, start_date, datetime.now())
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the 'datetime' format
rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')


def gpu_worker(in_qu: multiprocessing.Queue, out_qu: multiprocessing.Queue):
    exp = Exp_crossformer(args)
    model_dict = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location='cpu')
    exp.model.load_state_dict(model_dict)

    # load the data
    args.scale_statistic = pickle.load(open(os.path.join(args.checkpoint_dir, 'scale_statistic.pkl'), 'rb'))
    args.root_path = hyper_parameters['root_path']
    args.data_path = hyper_parameters['data_path']
    if args.different_split:
        data_split = string_split(args.data_split)
        args.data_split = data_split
    else:
        args.data_split = hyper_parameters['data_split']

    df_data = rates_frame[COLUMN_NAMES]
    scaler = StandardScaler(mean=args.scale_statistic['mean'], std=args.scale_statistic['std'])
    data = torch.Tensor(scaler.transform(df_data.values)).float().to(exp.device)
    exp.model.eval()
    # Not supported on Windows
    # exp.model = torch.compile(exp.model)

    while True:
        i_rate = in_qu.get()
        if i_rate is None:
            return
        with torch.no_grad():
            history = data.unfold(dimension=0, size=args.in_len, step=1)[i_rate:i_rate+BATCH_SIZE]
            history = history.permute(0, 2, 1)
            prediction = exp.model(history)
            prediction = scaler.inverse_transform(prediction)
            prediction = prediction.to('cpu').numpy()
            dfs_prediction = [pd.DataFrame(prediction[i, :, :], columns=COLUMN_NAMES) for i in range(prediction.shape[0])]
            out_qu.put({'i_rate': i_rate, 'predictions': dfs_prediction})


if __name__ == '__main__':
    processes = []
    n_gpus = torch.cuda.device_count()
    task_qu = multiprocessing.Queue()
    res_qu = multiprocessing.Queue()
    for i in range(n_gpus):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        process = multiprocessing.Process(target=gpu_worker, name='BacktesterGPU', args=(task_qu, res_qu))
        processes.append(process)
        process.start()
    del os.environ['CUDA_VISIBLE_DEVICES']

    pos_open = None
    pos_dir = 0
    capital = 1
    n_pos = 0
    n_days = 0

    n_items = rates_frame.shape[0] - args.out_len - args.in_len
    for i in range(0, N_BATCHES_AHEAD*BATCH_SIZE, BATCH_SIZE):
        task_qu.put(i)
    df_data = rates_frame[COLUMN_NAMES]
    gpu_res = dict()
    for i in tqdm(range(n_items)):
        if i%BATCH_SIZE == 0 and i+N_BATCHES_AHEAD*BATCH_SIZE < rates_frame.shape[0]-args.out_len:
            task_qu.put(i+N_BATCHES_AHEAD*BATCH_SIZE)
        if pos_open is not None:
            cur_pl = calc_pl(pos_open, open_price, pos_dir)
            if capital <= LEVERAGE * capital * cur_pl:
                print('Margin call')
                sys.exit(-1)
        if (i+1) % BARS_PER_DAY == 0:
            n_days += 1
            yearly_growth = math.pow(math.pow(capital, 1.0/n_days), 261)
            print(f'\tCapital: {capital:.5f}, n_pos: {n_pos}, n_days: {n_days}, yearly: {yearly_growth:.5f}')
        while gpu_res.get(i - i%BATCH_SIZE) is None:
            cur_res = res_qu.get()
            gpu_res[cur_res['i_rate']] = cur_res
        cur_res = gpu_res[i - i%BATCH_SIZE]
        if i % BATCH_SIZE == BATCH_SIZE-1:
            del gpu_res[i - i%BATCH_SIZE]

        i_rate = i + args.in_len
        df_history = df_data.iloc[i:i_rate].reset_index(drop=True)
        df_actual = df_data.iloc[i_rate:i_rate + args.out_len].reset_index(drop=True)
        df_prediction = cur_res['predictions'][i%BATCH_SIZE]
        open_price = df_history['close'].iloc[-1]
        min_price = df_prediction['low'].min()
        max_price = df_prediction['high'].max()
        delta_min = open_price / min_price
        delta_max = max_price / open_price
        delta_diff = math.fabs(delta_max - delta_min)
        if delta_max >= delta_min:
            open_dir = 1
        elif delta_max <= delta_min:
            open_dir = -1
        else:
            open_dir = 0
        b_close = False
        # if pos_dir > 0 and max_price <= df_actual['high'][0]:
        #     b_close = True
        # elif pos_dir < 0 and min_price >= df_actual['low'][0]:
        #     b_close = True
        if b_close:
            cur_pl = calc_pl(pos_open, open_price, pos_dir)
            capital += LEVERAGE * capital * cur_pl
            pos_dir = 0
            pos_open = None
        if  not pos_dir or (pos_dir * open_dir < 0 and delta_diff > MIN_GROWTH_PERC * 0.01):
            if pos_dir:
                cur_pl = calc_pl(pos_open, open_price, pos_dir)
                capital += LEVERAGE * capital * cur_pl
            pos_dir = open_dir
            n_pos += 1
            pos_open = open_price
    if pos_open is not None:
        cur_pl = calc_pl(pos_open, df_actual['close'].iloc[-1], pos_dir)
        capital += LEVERAGE * capital * cur_pl
    yearly_growth = math.pow(math.pow(capital, 1.0/n_days), 261)
    print(f'Total capital: {capital:.5f}, n_pos: {n_pos}, yearly: {yearly_growth:.5f}')

    # mae, mse, rmse, mape, mspe = exp.eval(args.setting_name, args.save_pred, args.inverse)
    for process in processes:
        task_qu.put(None)
    for process in processes:
        process.join()

mt5.shutdown()
