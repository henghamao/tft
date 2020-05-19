import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class StockFormatter(GenericDataFormatter):
    """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

    _column_definition = [
        ('code', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('close', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('open', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('high', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('low', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('amount', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
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
        ('code_hash', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

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
        test_ratio = 0.2
        valid_ratio = 0.1
        print('Test ratio:%f' % test_ratio)
        total_len = len(df)
        test_len = int(total_len * test_ratio)
        valid_len = int(total_len * valid_ratio)
        train_len = total_len - valid_len - test_len
        print(
            'Total size:%d, train size:%d, test size:%d, valid size:%d' % (total_len, train_len, test_len, valid_len))

        print('Formatting train-valid-test splits.')

        train = df.iloc[:train_len]
        valid = df.iloc[train_len + test_len:]
        test = df.iloc[train_len:train_len + test_len]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

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
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 40,
            'num_encoder_steps': 39,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'multiprocessing_workers': 5,
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
