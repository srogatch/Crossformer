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

c_spread = 1.5259520851045277178296601486713e-4
min_growth = math.pow(1 + c_spread, 10)
max_daily_growth = 1.06
minute_daily_growth = math.pow(max_daily_growth, 1.0/480)
LEVERAGE = 10
BARS_PER_DAY = 1440
DIRECTION = -1 # 1 for Buy, -1 for Sell
COLUMN_NAMES = ['open', 'high', 'low', 'close', 'tick_volume']


def calc_pl(open_price: float, close_price: float):
    if DIRECTION > 0:
        cur_pl = close_price / ((1 + c_spread) * open_price) - 1
    else:
        cur_pl = open_price / ((1 + c_spread) * close_price) - 1
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

    while True:
        i_rate = in_qu.get()
        if i_rate is None:
            return
        with torch.no_grad():
            history = data[i_rate - args.in_len:i_rate].unsqueeze(0)
            prediction = exp.model(history)
            prediction = scaler.inverse_transform(prediction).squeeze(0).to('cpu').numpy()
            df_prediction = pd.DataFrame(prediction, columns=COLUMN_NAMES)
            out_qu.put({'i_rate': i_rate, 'prediction': df_prediction})


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
    capital = 1
    n_pos = 0
    n_days = 0

    for i in range(args.in_len, args.in_len+4):
        task_qu.put(i)
    df_data = rates_frame[COLUMN_NAMES]
    growth_series = np.geomspace(
        start=min_growth, stop=min_growth*pow(minute_daily_growth, args.out_len-1), num=args.out_len)
    items = range(args.in_len, rates_frame.shape[0]-args.out_len)
    gpu_res = dict()
    for i in tqdm(items):
        if i+4 < rates_frame.shape[0]-args.out_len:
            task_qu.put(i+4)
        if (i+1) % BARS_PER_DAY == 0:
            n_days += 1
            yearly_growth = math.pow(math.pow(capital, 1.0/n_days), 261)
            print('Running capital:', capital, ' n_pos:', n_pos, ' n_days:', n_days, ' yearly:', yearly_growth)
        while gpu_res.get(i) is None:
            cur_res = res_qu.get()
            gpu_res[cur_res['i_rate']] = cur_res
        cur_res = gpu_res[i]
        del gpu_res[i]

        df_history = df_data.iloc[i - args.in_len:i].reset_index(drop=True)
        df_actual = df_data.iloc[i:i + args.out_len].reset_index(drop=True)
        df_prediction = cur_res['prediction']
        if pos_open is None:
            if df_history['close'].iloc[-1] <= df_prediction['low'][0]:
                buy_market = df_history['close'].iloc[-1]
                b_open = np.any(df_prediction['high'] >= growth_series * buy_market)
                if b_open:
                    pos_open = buy_market
                    n_pos += 1
                req_growth = min_growth
            else:
                buy_limit = df_prediction['low'][0]
                req_growth = min_growth
                b_enter = np.any(df_prediction['high'][1:] >= growth_series[:-1] * buy_limit)
                if b_enter and buy_limit >= df_actual['low'][0]:
                    pos_open = buy_limit
                    n_pos += 1
        else:
            if df_prediction['low'][0] * min_growth < df_history['close'].iloc[-1]:
                take_profit = df_prediction['high'][0]
                if (DIRECTION < 0 and df_actual['low'][0] <= take_profit) or \
                        (DIRECTION > 0 and df_actual['high'][0] >= take_profit):
                    cur_pl = calc_pl(pos_open, take_profit)
                    pos_open = None
                    capital += LEVERAGE * capital * cur_pl
                    if capital <= 0:
                        print('Margin call')
                        sys.exit(-1)
    if pos_open is not None:
        cur_pl = calc_pl(pos_open, df_actual['close'].iloc[-1])
        capital += LEVERAGE * capital * cur_pl
    yearly_growth = math.pow(math.pow(capital, 1.0/n_days), 261)
    print('Total capital:', capital, ' n_pos:', n_pos, ' yearly:', yearly_growth)

    # mae, mse, rmse, mape, mspe = exp.eval(args.setting_name, args.save_pred, args.inverse)
    for process in processes:
        task_qu.put(None)
    for process in processes:
        process.join()

mt5.shutdown()
