import sys
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from pprint import pprint

usage = 'PreProcessor.py /path/to/raw/dataset.csv /path/to/processed/dataset.csv'

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

    data = pd.read_csv(input_data_path, header=None)

    #clean missing and nulls
    data.dropna(axis=0, how='any')

    standard_scaler = StandardScaler()
    label_binarizer = LabelBinarizer()

    data.rename(index=str, columns={data.columns[-1]: 'classifier'}, inplace=True)

   #normalize data
    for column_name in data.columns:
        column = data[column_name]
        if is_numeric_dtype(column):
            data[column_name] = standard_scaler.fit_transform(data[[column_name]].as_matrix())
        elif is_string_dtype(column):
            encoded_array = label_binarizer.fit_transform(column)
            #remove old categorical columns
            data.drop(column_name, axis=1, inplace=True)
            #add newly encoded columns to dataframe
            new_columns_dict = {}
            #construct dictionary with keys of new column names (indicies starting at 1 past the last index of dataframe)
            # and values of the new encoded columns
            for i in range(0, encoded_array.shape[1]):
                encoded_column_name = str(column_name) + '=' + str(label_binarizer.classes_[i])
                column_data = encoded_array.T[i]
                new_columns_dict[encoded_column_name] = column_data
            #add new columns to dataframe
            data = data.assign(**new_columns_dict)
        elif not is_bool_dtype(column):
            raise TypeError('column is of unhandleable dtype: ', column.dtype)

    with output_data_path.open(mode='w') as output_data_stream:
        data.to_csv(output_data_stream, index=False)#, header=None)

    return 0

main(sys.argv)