import sys
import pandas as pd
from pathlib import Path

usage = 'PreProcessor.py /path/to/raw/dataset /path/to/processed/dataset'

if len(sys.argv) < 2:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 2:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)

def main():
    input_data_path = Path(sys.argv[1])
    output_data_path = Path(sys.argv[2])

    raw_input_data = pd.read_csv(input_data_path)


    return 0
