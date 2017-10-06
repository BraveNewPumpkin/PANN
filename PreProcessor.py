import sys
import pandas as pd
from pathlib import Path
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

    pprint(raw_input_data)

    return 0

main(sys.argv)