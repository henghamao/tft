import random

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import os
import pandas as pd
from data_formatters.data_model import StockDataSet

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


def load_stocks(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05, window=1, train=False,
                evaluate=False, from_db=True, data_dir='data'):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio,
                window=window,
                train=train,
                evaluate=evaluate,
                from_db=from_db,
                data_dir=data_dir,
            )
        ]

    symbols = []
    data_dir = os.path.join(data_dir, 'data')
    data_dir = os.path.join(data_dir, 'stock')
    file_black = ["_stock_list.csv", "stock_list.csv", "constituents-financials.csv"]
    # Load metadata
    s = dict()
    if os.path.exists(os.path.join(data_dir, 'stock_list.csv')):
        info = pd.read_csv(os.path.join(data_dir, 'stock_list.csv'))
        info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
        info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
        print(info['file_exists'].value_counts().to_dict())
        info = info[info['file_exists'] == True].reset_index(drop=True)
        symbols = info['symbol'].tolist()
    else:

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.csv' and file not in file_black:
                    s[os.path.splitext(file)[0]] = os.path.getsize(os.path.join(data_dir, file))
            # Order by file size (data samples)
            symbols += sorted(s.items(), key=lambda d: d[1], reverse=True)
        symbols = [symbols[i][0] for i in range(len(symbols))]

    if k is not None:
        symbols = symbols[:k]
    sl = []
    p = 0
    for s in symbols:
        p += 1
        print('Init data for symbol:%s, progress: %d/%d' % (s, p, len(symbols)))
        sl.append(StockDataSet(s,
                               input_size=input_size,
                               num_steps=num_steps,
                               test_ratio=test_ratio,
                               window=window,
                               train=train,
                               evaluate=evaluate,
                               data_dir=data_dir,
                               ))
    return sl


class StockFormatter(GenericDataFormatter):
    """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

    _column_definition = [
        ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('close', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('open', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('high', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('low', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('turn', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('sma5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('code_hash', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('circulating_market_cap', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pe_ratio', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pb_ratio', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ps_ratio', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pcf_ratio', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('net_pct_main', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('net_pct_xl', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('net_pct_l', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('net_pct_m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('net_pct_s', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        # the number of csv files to read from the data folder
        self.stock_count = 10
        self.input_size = 22
        self.fix_param = self.get_fixed_params()
        self.num_steps = self.fix_param['total_time_steps']
        self.sl = load_stocks(input_size=self.input_size, k=self.stock_count, num_steps=self.num_steps, train=True)

    def split_data(self, df, valid_boundary=2016, test_boundary=2018):
        """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

        stock_count = len(self.sl)
        test_ratio = 0.2
        print('Stock count:%d'% stock_count)
        train_x = []
        test_x = []
        for label_, d_ in enumerate(self.sl):
            stock_train_len = int(len(d_.train_y) * (1 - test_ratio))
            train_x += list(d_.train_x[:stock_train_len])
            test_x += list(d_.train_x[stock_train_len:])

        train_g = pd.DataFrame(train_x, columns=([k[0] for k in self._column_definition]))
        test_g = pd.DataFrame(test_x, columns=([k[0] for k in self._column_definition]))

        self.set_scalers(train_g)

        def tofloat(data):
            for col in data.columns:
                if col not in {'Symbol', 'date'}:
                    data[col] = data[col].astype('float32')
            return data


        train_g = tofloat(train_g)
        test_g = tofloat(test_g)
        # used test for both valid and test
        return train_g, test_g, test_g

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = predictions[col]

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 40,
            'num_encoder_steps': 39,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'multiprocessing_workers': 2,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 160,
            'learning_rate': 0.01,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 1,
            'stack_size': 1
        }

        return model_params
