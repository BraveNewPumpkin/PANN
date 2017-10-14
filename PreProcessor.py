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

    data = pd.read_csv(input_data_path, header=None)

    #clean missing and nulls
    data.dropna(axis=0, how='any')

    standard_scaler = StandardScaler()
    label_encoder = LabelEncoder()

    #temporarily save and remove classifier column from dataframe
    classifier_name = data.columns[-1]
    classifier_column = data[classifier_name]
    data.drop(classifier_name, axis=1, inplace=True)


    #normalize data
    categorical_column_names = []
    for column_name in data.columns:
        column = data[column_name]
        if is_numeric_dtype(column):
            data[column_name] = standard_scaler.fit_transform(data[[column_name]].as_matrix())
        elif is_string_dtype(column):
            data[column_name] = label_encoder.fit_transform(data[column_name])
            categorical_column_names.append(column_name)
        elif not is_bool_dtype(column):
            raise TypeError('column is of unhandleable dtype: ', column.dtype)

    one_hot_encoder = OneHotEncoder(categorical_features=categorical_column_names, sparse=False)
    encoded_array = one_hot_encoder.fit_transform(data)

    #remove old categorical columns
    data.drop(categorical_column_names, axis=1, inplace=True)
    #add newly encoded columns to dataframe
    new_columns_dict = {}
    #construct dictionary with keys of new column names (indicies starting at 1 past the last index of dataframe)
    # and values of the new encoded columns
    for i in range(0, encoded_array.shape[1]):
        column_name = str(data.shape[1] + i + 1)
        column_data = encoded_array.T[i]
        new_columns_dict[column_name] = column_data
    #add new columns to dataframe
    data = data.assign(**new_columns_dict)

    #add classifier column back on
    data.insert(loc=data.shape[1], column='classifier', value=classifier_column)

    with output_data_path.open(mode='w') as output_data_stream:
        data.to_csv(output_data_stream)

    return 0

main(sys.argv)