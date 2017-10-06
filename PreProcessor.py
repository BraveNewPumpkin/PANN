import sys
import pandas as pd
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

@unique
class ColumnType(IntEnum):
    REAL = auto()
    CATEGORICAL = auto()

def main(argv):
    input_data_path = Path(argv[1])
    output_data_path = Path(argv[2])

    raw_input_data = pd.read_csv(input_data_path, header=None)

    #TODO clean missing and nulls

    for column_name in raw_input_data:
        column_type = guessColType(raw_input_data, column_name)
        print(column_type)

    return 0


def guessColType(raw_input_data, column_name):
    column_type = None
    unique_values = raw_input_data[column_name].unique()
    pprint(unique_values)
    #print('column: %s -> values: %s' % (column_name, ', '.join(unique_values)))
    try:
        for value in unique_values:
            float(value)
        column_type = ColumnType.REAL
    except ValueError:
        column_type = ColumnType.CATEGORICAL

    return column_type

main(sys.argv)