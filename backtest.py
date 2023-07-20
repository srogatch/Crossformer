import argparse
import os
import pickle
from datetime import datetime
import sys

import pandas as pd
import MetaTrader5 as mt5
import torch

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import load_args, string_split, StandardScaler

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    sys.exit(1)

rates = mt5.copy_rates_range('US500.pro', mt5.TIMEFRAME_M1, datetime(2022, 4, 8, 1, 37), datetime.now())
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the 'datetime' format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')

parser = argparse.ArgumentParser(description='CrossFormer Strategy Backtester')
parser.add_argument('--checkpoint_root', type=str, default='./checkpoints', help='location of the trained model')
parser.add_argument(
    '--setting_name', type=str, default='Crossformer_prepared-US500_M1_il1440_ol480_sl6_win2_fa10_dm1024_nh16_el6_itr0',
    help='name of the experiment')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--different_split', action='store_true', help='use data split different from training process', default=False)
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='data split of train, vali, test')

parser.add_argument('--inverse', action='store_true', help='inverse output data into the original scale', default=False)
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False

args.checkpoint_dir = os.path.join(args.checkpoint_root, args.setting_name)
hyper_parameters = load_args(os.path.join(args.checkpoint_dir, 'args.json'))

#load the pre-trained model
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
exp = Exp_crossformer(args)
model_dict = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location='cpu')
exp.model.load_state_dict(model_dict)

#load the data
args.scale_statistic = pickle.load(open(os.path.join(args.checkpoint_dir, 'scale_statistic.pkl'), 'rb'))
args.root_path = hyper_parameters['root_path']
args.data_path = hyper_parameters['data_path']
if args.different_split:
    data_split = string_split(args.data_split)
    args.data_split = data_split
else:
    args.data_split = hyper_parameters['data_split']

column_names = ['open', 'high', 'low', 'close', 'tick_volume']
df_data = rates_frame[column_names]
scaler = StandardScaler(mean = args.scale_statistic['mean'], std = args.scale_statistic['std'])
data = scaler.transform(df_data.values)
exp.model.eval()

with torch.no_grad():
    for i in range(args.in_len, data.shape[0]-args.out_len):
        history = torch.Tensor(data[i-args.in_len:i]).unsqueeze(0).float().to(exp.device)
        actual = torch.Tensor(data[i:i+args.out_len]).unsqueeze(0).float().to(exp.device)
        prediction = exp.model(history)
        history = scaler.inverse_transform(history).squeeze(0).to('cpu').numpy()
        actual = scaler.inverse_transform(actual).squeeze(0).to('cpu').numpy()
        prediction = scaler.inverse_transform(prediction).squeeze(0).to('cpu').numpy()
        df_history = pd.DataFrame(history, columns=column_names)
        df_actual = pd.DataFrame(actual, columns=column_names)
        df_prediction = pd.DataFrame(prediction, columns=column_names)


# mae, mse, rmse, mape, mspe = exp.eval(args.setting_name, args.save_pred, args.inverse)

mt5.shutdown()
