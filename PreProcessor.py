import sys
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
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

    raw_input_data = pd.read_csv(input_data_path, header=None)

    #clean missing and nulls
    raw_input_data.dropna(axis=0, how='any')

    standard_scaler = StandardScaler()
    label_encoder = LabelEncoder()

    #normalize data
    categorical_column_names = []
    for column_name in raw_input_data.columns:
        column = raw_input_data[column_name]
        if is_numeric_dtype(column):
            raw_input_data[column_name] = standard_scaler.fit_transform(raw_input_data[[column_name]].as_matrix())
        elif is_string_dtype(column):
            raw_input_data[column_name] = label_encoder.fit_transform(raw_input_data[column_name])
            categorical_column_names.append(column_name)
        elif not is_bool_dtype(column):
            raise TypeError('column is of unhandleable dtype: ', column.dtype)

    one_hot_encoder = OneHotEncoder(categorical_features=categorical_column_names, sparse=False)
    encoded_array = one_hot_encoder.fit_transform(raw_input_data)

    #remove old categorical columns
    raw_input_data.drop(categorical_column_names, axis=1, inplace=True)
    #add newly encoded columns to dataframe
    new_columns_dict = {}
    #construct dictionary with keys of new column names (indicies starting at 1 past the last index of dataframe)
    # and values of the new encoded columns
    for i in range(0, encoded_array.shape[1]):
       new_columns_dict[raw_input_data.shape[1] + i + 1] = encoded_array.T[i]
    pprint(new_columns_dict)
    #add new columns to dataframe
    raw_input_data.assign(new_columns_dict)

#    raw_input_data.to_csv(output_data_path)

    return 0

main(sys.argv)