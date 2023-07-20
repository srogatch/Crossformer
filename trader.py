import argparse
import math
import multiprocessing
import os
import pickle
from datetime import datetime
import sys

import pandas as pd
import MetaTrader5 as mt5
import torch
from tqdm import tqdm

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import load_args, string_split, StandardScaler

c_spread = 1.5259520851045277178296601486713e-4
min_growth = math.pow(1 + c_spread, 3)
max_daily_growth = 1.06
minute_daily_growth = math.pow(max_daily_growth, 1.0/480)
LEVERAGE = 10
BARS_PER_DAY = 1440

