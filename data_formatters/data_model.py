import datetime
import gc
import math

import numpy as np
import os
import pandas as pd
import random
import time
import sqlalchemy
import pandas_ta
import datetime

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=4,
                 num_steps=30,
                 normalized=True,
                 window=1,
                 test_ratio=0.2,
                 train=True,
                 evaluate=False,
                 from_db=False,
                 data_dir='data',
                 start_date='',
                 end_date=''):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.normalized = normalized
        self.window = window
        self.noise = False
        self.code = self.myhash(self.stock_sym)
        self.small = False

        raw_df = pd.read_csv(os.path.join(data_dir, "%s.csv" % stock_sym))
        raw_df.columns = [x.lower() for x in raw_df.columns]
        if not 'close' in raw_df:
            raise Exception('Not valid close price in data ' + os.path.join("data", "%s.csv" % stock_sym))
        delta = num_steps
        if self.input_size >= 7:
            # sma5
            delta += 5
        if start_date != '':
            if not start_date.__contains__('-'):
                start_date = datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
            if evaluate:
                if len(raw_df[raw_df.date > start_date].index) == 0:
                    self.small = True
                    print('Warn: No matching data with %s for the stock %s.' % (start_date, stock_sym))
                    return
                index = raw_df[raw_df.date >= start_date].index[0] - delta
                index = index if index > 0 else 0
                raw_df = raw_df.loc[index:].reset_index(drop=True)
            else:
                raw_df = raw_df[raw_df['date'] >= start_date].reset_index(drop=True)
        if end_date != '':
            if not end_date.__contains__('-'):
                end_date = datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            raw_df = raw_df[raw_df['date'] <= end_date].reset_index(drop=True)
        if (len(raw_df) < delta + 1 and not train) or \
                (len(raw_df) < delta + 2 and evaluate) or \
                (int(len(raw_df) * (1 - test_ratio)) < delta + window + 1 and train):
            print("Warn: %s sample is too small after date filtering, length %d." % (stock_sym, len(raw_df)))
            self.small = True
            return
        if not train and not evaluate:
            # only last samples are needed to predict the next value, one more value for normalized
            if self.input_size >= 7:
                # +5 for SMA5 calculation
                raw_df = raw_df.tail(num_steps + 5).reset_index(drop=True)
            else:
                raw_df = raw_df.tail(num_steps + 1).reset_index(drop=True)
        if not self.check(raw_df):
            self.noise = True
            return
        # Merge into one sequence
        if self.input_size == 22:
            # Extract features: Close price, Open, High, Low, Volume, turn, sma5, code_hash, cash_flow(5 features),
            # fin_data(6 features), date_features(4)
            sma5 = raw_df.ta.sma(length=5)
            sma5 = sma5.tail(-4).reset_index(drop=True)
            raw_df = raw_df.tail(-4).reset_index(drop=True)
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn'), s, getattr(row, 'net_pct_main'),
                             getattr(row, 'net_pct_xl'), getattr(row, 'net_pct_l'), getattr(row, 'net_pct_m'),
                             getattr(row, 'net_pct_s'), getattr(row, 'circulating_market_cap'),
                             getattr(row, 'pe_ratio'), getattr(row, 'pb_ratio'), getattr(row, 'ps_ratio'),
                             getattr(row, 'pcf_ratio')] for (row, s) in zip(raw_df.itertuples(), sma5)]
            self.date_f = [self.date_features(getattr(row, 'date')) for row in raw_df.itertuples()]
            self.date = [getattr(row, 'date') for row in raw_df.itertuples()]
        elif self.input_size == 18:
            # Extract features: Close price, Open, High, Low, Volume, turn, sma5, code_hash, cash_flow(5 features),
            # fin_data(6 features)
            sma5 = raw_df.ta.sma(length=5)
            sma5 = sma5.tail(-4).reset_index(drop=True)
            raw_df = raw_df.tail(-4).reset_index(drop=True)
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn'), s, getattr(row, 'net_pct_main'),
                             getattr(row, 'net_pct_xl'), getattr(row, 'net_pct_l'), getattr(row, 'net_pct_m'),
                             getattr(row, 'net_pct_s'), getattr(row, 'circulating_market_cap'),
                             getattr(row, 'pe_ratio'), getattr(row, 'pb_ratio'), getattr(row, 'ps_ratio'),
                             getattr(row, 'pcf_ratio')] for (row, s) in zip(raw_df.itertuples(), sma5)]
        elif self.input_size == 13:
            # Extract features: Close price, Open, High, Low, Volume, turn, sma5, code_hash, cash_flow(5 features)
            sma5 = raw_df.ta.sma(length=5)
            sma5 = sma5.tail(-4).reset_index(drop=True)
            raw_df = raw_df.tail(-4).reset_index(drop=True)
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn'), s, getattr(row, 'net_pct_main'),
                             getattr(row, 'net_pct_xl'), getattr(row, 'net_pct_l'), getattr(row, 'net_pct_m'),
                             getattr(row, 'net_pct_s')] for (row, s) in zip(raw_df.itertuples(), sma5)]
        elif self.input_size == 8:
            # Extract features: Close price, Open, High, Low, Volume, turn, sma5, code_hash
            sma5 = raw_df.ta.sma(length=5)
            sma5 = sma5.tail(-4).reset_index(drop=True)
            raw_df = raw_df.tail(-4).reset_index(drop=True)
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn'), s] for (row, s) in zip(raw_df.itertuples(),
                                                                                                  sma5)]
        elif self.input_size == 7:
            # Extract features: Close price, Open, High, Low, Volume, turn, sma5
            sma5 = raw_df.ta.sma(length=5)
            sma5 = sma5.tail(-4).reset_index(drop=True)
            raw_df = raw_df.tail(-4).reset_index(drop=True)
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn'), s] for (row, s) in
                            zip(raw_df.itertuples(), sma5)]
        elif self.input_size == 6:
            # Extract features: Close price, Open, High, Low, Volume, turn
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume'), getattr(row, 'turn')] for row in raw_df.itertuples()]
        elif self.input_size == 5:
            # Extract features: Close price, Open, High, Low, Volume
            self.raw_seq = [[getattr(row, 'close'), getattr(row, 'open'), getattr(row, 'high'), getattr(row, 'low'),
                             getattr(row, 'volume')] for row in raw_df.itertuples()]
        elif self.input_size == 1:
            # Extract feature: Close price
            self.raw_seq = [[getattr(row, 'close')] for row in raw_df.itertuples()]
        else:
            raise Exception('Not valid input_size:%d' % self.input_size)

        if evaluate:
            self.predict_x, self.evaluate_y = self.prepare_data_evaluate(self.raw_seq, window)
            if window > 1:
                self.evaluate_x = self.predict_x[:1 - window]
            else:
                self.evaluate_x = self.predict_x
            # for evaluation, used window=1
            self.date = raw_df['date'].tolist()[num_steps + 1:]
            self.close_price = raw_df['close'].tolist()[num_steps + 1:]
            self.open_price = raw_df['open'].tolist()[num_steps + 1:]
        elif train:
            self.train_x, self.train_y = self.prepare_data(self.raw_seq, window)
        else:
            self.predict_x = self.prepare_data_predict(self.raw_seq)
            self.close_price = []
            self.date = raw_df['date'].tail(1).to_list()
        # memory optimization
        del raw_df
        del self.raw_seq
        gc.collect()

    def check(self, df):
        if df.isnull().any().any():
            return False
        close = df['close']
        for i in range(2, len(df)):
            if abs(close[i] / close[i - 1] - 1) > 0.15:
                return False
        # fix 0 volume
        df['volume'] = df['volume'].replace(0, 1)
        return True

    def myhash(self, s):
        # the new function avoid the hash conflict
        # Test result on 02/06: my hash conflict: 0.38; my hash2 conflict 0
        bucket_size = 19623
        t = s.replace('sh.', '1')
        t = t.replace('sz.', '5')
        if not t.isdigit():
            t = t.__hash__()
        return int(t) % bucket_size / bucket_size

    def prepare_data(self, seq, window=1):
        seq = np.array([np.array(seq[i]) for i in range(len(seq))])

        if self.normalized:
            y_seq = [[seq[i][0] / seq[i - window][0] - 1.0] for i in range(self.num_steps + window, len(seq))]
            seq = self.normalized_seq(seq)
        else:
            y_seq = [[seq[i]] for i in range(self.num_steps + window, len(seq))]
        # split into groups of num_steps
        # x = np.array([seq[i: i + self.num_steps] for i in range(0, len(seq) - self.num_steps - window + 1)])
        x = np.array([seq[i] for i in range(0, len(seq) - window + 1)])
        y = np.array([y for y in y_seq], dtype=np.float32)
        return x, y

    def prepare_data_evaluate(self, seq, window=1):
        seq = np.array([np.array(seq[i]) for i in range(len(seq))])

        if self.normalized:
            y_seq = [[seq[i][0] / seq[i - window][0] - 1.0] for i in range(self.num_steps + window, len(seq))]
            seq = self.normalized_seq(seq)
        else:
            y_seq = [[seq[i]] for i in range(self.num_steps + window, len(seq))]
        # split into groups of num_steps
        # x = np.array([seq[i: i + self.num_steps] for i in range(0, len(seq) - self.num_steps - window + 1)])
        # Note: len(x) != len(y). y used window size, while x used window=1
        # x = np.array([seq[i: i + self.num_steps] for i in range(0, len(seq) - self.num_steps)])
        x = np.array([x for x in seq], dtype=np.float32)
        y = np.array([y for y in y_seq], dtype=np.float32)
        return x, y

    def prepare_data_predict(self, seq):
        # split into items of input_size
        seq = np.array([np.array(seq[i]) for i in range(len(seq))])
        if self.normalized:
            seq = self.normalized_seq(seq)
        return np.array([seq], dtype=np.float32)

    def normalized_seq(self, seq):
        cap_norm = 1000
        if self.input_size == 1 or self.input_size == 2:
            # 1:close; 2:close, volume
            seq = [curr / seq[i] - 1.0 for i, curr in enumerate(seq[1:])]
        elif self.input_size == 5:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0] for i, curr in enumerate(seq[1:])]
        elif self.input_size == 6:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100]
                   for i, curr in enumerate(seq[1:])]
        elif self.input_size == 7:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn, sma5
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100, curr[6] / seq[i][6] - 1.0]
                   for i, curr in enumerate(seq[1:])]
        elif self.input_size == 8:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn, sma5, stock_sym
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100, curr[6] / seq[i][6] - 1.0,
                    self.code] for i, curr in enumerate(seq[1:])]
        elif self.input_size == 13:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn, sma5, stock_sym, cash_flow(5)
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100, curr[6] / seq[i][6] - 1.0,
                    self.code, curr[7], curr[8], curr[9], curr[10], curr[11]] for i, curr in enumerate(seq[1:])]
        elif self.input_size == 18:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn, sma5, stock_sym, cash_flow(5),
            # fin_data(5)
            seq = [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100, curr[6] / seq[i][6] - 1.0,
                    self.code, curr[7], curr[8], curr[9], curr[10], curr[11], curr[12] / cap_norm, 1 / curr[13],
                    1 / curr[14], 1 / curr[15], 1 / curr[16]]
                   for i, curr in enumerate(seq[1:])]
        elif self.input_size == 22:
            # Close/Close, Open/Close, High/Close, Low/Close, Volume/Volume, turn, sma5, stock_sym, cash_flow(5),
            # fin_data(5), date_features(4)
            # Added code and date
            seq = [[self.stock_sym, self.date[i+1], curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0,
                    curr[3] / seq[i][0] - 1.0, curr[4] / seq[i][4] - 1.0, curr[5] / 100, curr[6] / seq[i][6] - 1.0,
                    self.code, curr[7], curr[8], curr[9], curr[10], curr[11], curr[12] / cap_norm, 1 / curr[13],
                    1 / curr[14], 1 / curr[15], 1 / curr[16]] + self.date_f[i+1]
                   for i, curr in enumerate(seq[1:])]
        else:
            raise Exception('Not valid input_size:%d' % self.input_size)
        return seq

    @staticmethod
    def x_generator(x, batch_size, num_steps):
        while True:
            start = 0
            while start + batch_size < len(x) - num_steps:
                # Drop the last remaining data
                offset = start + batch_size
                batch_x = np.array([x[i:i + num_steps] for i in range(start, offset)], dtype=np.float32)
                # print('progress: %d/%d.' % (offset, len(x)))
                start = start + batch_size
                yield batch_x
            if start < len(x) - num_steps <= start + batch_size:
                offset = len(x) - num_steps
                batch_x = np.array([x[i:i + num_steps] for i in range(start, offset)], dtype=np.float32)
                yield batch_x

    @staticmethod
    def train_data_generator(x, y, batch_size, num_step, index, visit_seq, train_len):
        batch_count = math.ceil(train_len / batch_size)
        batch_seq = list(range(batch_count))
        while True:
            for batch in batch_seq:
                step = 0
                start = batch * batch_size
                batch_x = []
                batch_y = []
                while step < batch_size and start + step < train_len:
                    seq = visit_seq[start + step]
                    batch_y += [y[seq]]
                    seq = index[seq]
                    batch_x += [x[seq: seq + num_step]]
                    step += 1
                yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    @staticmethod
    def train_data(x, y, num_step, index, visit_seq, train_len):
        step = 0
        start = 0
        batch_x = []
        batch_y = []
        while step < train_len:
            seq = visit_seq[start + step]
            batch_y += [y[seq]]
            seq = index[seq]
            batch_x += [x[seq: seq + num_step]]
            step += 1
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    @staticmethod
    def test_data_generator(x, y, batch_size, num_step, index, visit_seq, train_len):
        count = len(y)
        test_len = count - train_len
        batch_count = math.ceil(test_len / batch_size)
        batch_seq = list(range(batch_count))
        while True:
            for batch in batch_seq:
                step = 0
                start = train_len + batch * batch_size
                batch_x = []
                batch_y = []
                while step < batch_size and start + step < count:
                    seq = visit_seq[start + step]
                    batch_y += [y[seq]]
                    seq = index[seq]
                    batch_x += [x[seq: seq + num_step]]
                    step += 1
                yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    @staticmethod
    def test_data(x, y, num_step, index, visit_seq, train_len):
        count = len(y)
        step = 0
        start = train_len
        batch_x = []
        batch_y = []
        while start + step < count:
            seq = visit_seq[start + step]
            batch_y += [y[seq]]
            seq = index[seq]
            batch_x += [x[seq: seq + num_step]]
            step += 1
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    @staticmethod
    def one_batch_test(d, batch_size, num_steps):
        x = d.train_x
        y = d.train_y
        batch = batch_size if len(d.train_y) > batch_size else len(d.train_y)
        batch_x = [x[i: i + num_steps] for i in range(len(d.train_y) - batch, len(d.train_y))]
        batch_y = [y[i] for i in range(len(d.train_y) - batch, len(d.train_y))]

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    @staticmethod
    def date_features(d):
        datetime.datetime.today()
        d = datetime.datetime.strptime(d, "%Y-%m-%d").date()
        day_of_week = d.weekday()
        day = d.day - 1
        month = d.month - 1
        week = d.isocalendar()[1] - 1
        return [day_of_week, day, week, month]
        # normalized by the category number
        #return [day_of_week / 7, day / 31, week / 53, month / 12]