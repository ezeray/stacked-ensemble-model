# -*- coding: utf-8 -*-
import logging
from datetime import datetime as dt
from numpy import ndarray
from pandas.core.frame import DataFrame as DF
from pandas.core.series import Series as Ser
import scipy.sparse.csr as csr


def create_logger(name, out_path='.'):
    """Create logger instance.

    Args:
        name (str): File and logger name

    Returns:
        logger: logger instance
    """
    # create logger instance
    log = logging.getLogger(f'{out_path}/{name}')
    log.setLevel(logging.DEBUG)
    now = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    # create file handler
    fh = logging.FileHandler(f'{name}_{now}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # format
    my_fmt = logging.Formatter(
        '%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s'
    )
    fh.setFormatter(my_fmt)
    ch.setFormatter(my_fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def convert_time(time):
    """
    Takes a measure in seconds and converts to the most appropriate
    format.
    Returns a string in float format with two decimals
    """
    if not (isinstance(time, int) or isinstance(time, float)):
        raise TypeError('time has to be either int or float')
    if time < 60:
        return f'{time:02.2f} seconds'
    elif time < 60*60:
        minutes = time // 60
        seconds = time % 60
        return f'{minutes:02.0f}:{seconds:02.2f} minutes'
    else:
        seconds = time % 60
        time_left = time // 60
        minutes = time_left % 24
        hours = time_left // 24
        return f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.2f} hours'


def check_input_data_type(data, one_dim=False):
    if isinstance(data, Ser):
        return data.values.reshape(-1)
    if isinstance(data, DF):
        return data.values
    if isinstance(data, ndarray):
        if one_dim:
            return data.reshape(-1)
        else:
            return data
    if isinstance(data, csr.csr_matrix):
        return data
    else:
        raise TypeError(
            'Data should be either a numpy array, a pandas '
            'dataframe/series, or a scipy sparse matrix.'
        )
