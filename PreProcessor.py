import sys
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
import numpy as np
from pathlib import Path
from enum import IntEnum, auto, unique
from pprint import pprint

usage = 'PreProcessor.py /path/to/raw/dataset /path/to/processed/dataset'

if len(sys.argv) < 3:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 3:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)

def main(argv):
    input_data_path = Path(argv[1])
    output_data_path = Path(argv[2])

    raw_input_data = pd.read_csv(input_data_path, header=None)

    #clean missing and nulls
    raw_input_data.dropna(axis=0, how='any')

    for column_name in raw_input_data.columns:
        column = raw_input_data[column_name]
        if is_numeric_dtype(column):
            #numericColumnHandler(raw_input_data, column_name)
            numericColumnHandler(column)
        elif is_string_dtype(column):
            stringColumnHandler(column)
        elif not is_bool_dtype(column):
            raise TypeError('column is of unhandleable dtype: ', column.dtype)


    return 0

def numericColumnHandler(column):
    print('numeric')

def stringColumnHandler(column):
    print('string')

main(sys.argv)