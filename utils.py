import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

from tqdm import tqdm
from glob import glob

import sys

from sklearn.preprocessing import LabelEncoder, Imputer,OneHotEncoder, StandardScaler

def get_descriptors(data, max_categories=10, nan_treshold=0.75, verbose=False):
    """
    
    gets the list with names of the columns that contain catagorical or numerical data
    
    nan_treshold: threshold (>0, <1) all numerical categories have to have this fraction of non-nan values to be considered
    max_categories: number of categories to be considered a category otherwise its considered to be a numerical value
    
    """
    descriptors_binary, descriptors_numeric, descriptors_categorical = [], [], []

    dropcount = 0
    for key in data.keys():

        if len(data[key].unique()) == 1:
            print('drop', key) # drop the ones that have only one unique value
            dropcount += 1
        elif len(data[key].unique()) == 2:
            descriptors_binary.append(key)
#             if verbose:
#                 print('binary', key, len(data[key].unique()))
        elif len(data[key].unique()) > max_categories and not np.all([isinstance(s, str) for s in data[key].unique()]):
            descriptors_numeric.append(key)
#             if verbose:
#                 print('numeric', key, len(data[key].unique()))
        else:
            descriptors_categorical.append(key)
#             if verbose:
#                 print('categorical', key, len(data[key].unique()))

    print('dropped ', dropcount, ' colunms')

#     descriptors_categorical = ['SK_ID_CURR'] + descriptors_categorical  # add ID to categorical
#     descriptors_binary = ['SK_ID_CURR'] + descriptors_binary  # add ID to binary
    if 'TARGET' in descriptors_binary:
        descriptors_binary.remove('TARGET') # remove target since this is not an input
        print('dropped TARGET because this is not a feature')
    if 'TARGET' in descriptors_categorical:
        descriptors_categorical.remove('TARGET') # remove target since this is not an input
        print('dropped TARGET because this is not a feature')
    
    if verbose:
        print('descriptors_binary: ', len(descriptors_binary))
        print('descriptors_numeric: ', len(descriptors_numeric))
        print('descriptors_categorical (all): ', len(descriptors_categorical))
        
    # drop the numerical categories where the number of nan value is too high
    descriptors_numeric = [desc for desc in descriptors_numeric if data[desc].count() / len(data) > nan_treshold]
    
    if verbose:
        print('descriptors_numeric (after threshold): ', len(descriptors_numeric))
    
    if verbose:
        print('descriptors_categorical (final): ', len(descriptors_categorical))
    return descriptors_binary, descriptors_categorical, descriptors_numeric

def drop_single_values(data_in, ignore_nan=True):


    def is_single_value(unique):
        """


        returns True if unique only has a single value
        if ignore_nan is True, then nans are not counted,
        e.g. if unique only has a single value and nan this will be considered only a single value
        Args:
            unique:

        Returns:

        """


        if len(unique)==1:
            return True

        if ignore_nan:
            if len(unique) == 2:

                if pd.isnull(unique).any():
                    return True

        return False


    
    single_value_keys = [key for key in data_in.columns if is_single_value(data_in[key].unique())]
    
    for key in single_value_keys:
        data_in.drop([key], axis=1, inplace=True)

def one_hot_encoder(df, nan_as_category = True, remove_single_values = True):
    """

    one hot encoder, doesn't work for numerical values (not of object type)


    Args:
        df:
        nan_as_category:
        remove_single_values:

    Returns:

    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    
    if remove_single_values:
        drop_single_values(df)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def outliers_iqr(ys, outlier_range=1.5):
    """
    finds outliers according to the interquartile range method
    see http://colingorrie.github.io/outlier-detection.html
    Args:
        ys:
        outlier_range: changes the sensitiviy to outliers


    Returns: indecies of outliers

    """
    quartile_1, quartile_3 = np.percentile(ys[np.isfinite(ys)], [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * outlier_range)
    upper_bound = quartile_3 + (iqr * outlier_range)
    indecies =  np.where((ys > upper_bound) | (ys < lower_bound))[0]

    outlier = ys.iloc[indecies]
    return outlier

def nunique(array,return_counts=True):
    """

    wrapper for np.unique that ignores nan
    Returns:

    """

    if return_counts:
        u, c = np.unique(array, return_counts=True)

        u, c = u[np.isfinite(u)], c[np.isfinite(u)]
        return u, c
    else:
        u = np.unique(array)
        # u = list(u[np.isfinite(u)]) + [np.nan]
        u = u[np.isfinite(u)]
        return u




def get_descriptors_outlier(data,  outlier_range=1.5, outlier_fraction=0.5):
    """

    Args:
        data:
        outlier_factor:
        outlier_fraction: consider only outliers that are at least this fraction of the total data

    Returns:

    """
    descriptors_numeric_outlier = [key for key in data.columns
                                   if get_outlier_fraction(data[key], outlier_range) > outlier_fraction]

    return descriptors_numeric_outlier

def get_outlier_fraction(y, outlier_range=1.5):
    """


    Args:
        y:

    Returns:

    """
    outlier = outliers_iqr(y, outlier_range=outlier_range)
    return len(outlier)/len(y)


def get_outlier_info(data, outlier_range=1.5, verbose=False):
    """


    Args:
        data: datafame the will be analysed for outliers
        outlier_range:
        verbose:

    Returns:

    """


    return_dict = {
        'column': [],
        'value': [],
        'freq. ratio (mean)': [],'freq. ratio (next)': [],
        'percent': [],
        'counts': [],
        'unique':[], 'max':[]
    }
    for key in data.columns:

        if verbose:
            print(key)
        outliers = outliers_iqr(data[key], outlier_range=outlier_range)

        unique, counts = nunique(outliers, return_counts=True)
        if verbose:
            print('found', len(outliers), 'outliers in ', len(data[key]), 'values')
            print(len(unique), 'are unique ', unique)
        unique = [x for _, x in sorted(zip(counts, unique), key=lambda pair: pair[0])]
        counts = sorted(counts)
        if verbose:
            print('max count', max(counts))
        if verbose:
            print('the mose frequent value', unique[-1], 'is ', counts[-1] / np.mean(counts),
                  'more frequent than the mean')
            print('and represents ', counts[-1] / len(data) * 100, 'percent of all values ')
        return_dict['column'].append(key)
        return_dict['value'].append(unique[-1])
        return_dict['freq. ratio (mean)'].append(counts[-1] / np.mean(counts))
        return_dict['freq. ratio (next)'].append(counts[-1] / counts[-2])
        return_dict['percent'].append(counts[-1] / len(data[key]))
        return_dict['counts'].append(counts[-1])
        return_dict['unique'].append(len(unique))
        return_dict['max'].append(max(counts))

    df_info = pd.DataFrame.from_dict(return_dict)
    df_info.sort_values(by=['freq. ratio (next)', 'percent'], ascending=False, inplace=True)

    return df_info

        
def clean_nan(data_in, aux_nan_numeric = -999999):
    """
    if unique values of data_in are all str exept for nan value, replace nan with str N/A
    if unique values of data_in are all numeric, replace nan with aux_nan_numeric, since numpy.unique fails on nan 
    (i.e. gives each nan value as a seperate value)
    
    aux_nan_numeric
    
    """
    
    # get all categories that are not nan
    categories_no_nan = list(data_in.unique())
    if np.nan in categories_no_nan:
        categories_no_nan.remove(np.nan) # remove nan 
        
    # if all the non nan categories are str convert nan to N/A string
    if all(isinstance(x, str) for x in categories_no_nan):
        data_in = data_in.fillna('N/A')
        
    elif all(isinstance(x, (float, np.float, int, np.int)) for x in categories_no_nan):
        if aux_nan_numeric in categories_no_nan:
            print('WARNING aux_nan_numeric in categories')
#         data_in.fillna(aux_nan_numeric, inplace=True)
        data_in = data_in.fillna(aux_nan_numeric)
#     print('asdsad', categories_no_nan)

    return data_in



def categorical_to_onehot(data_in):
    """
    takes in a pandas df with first column SK_ID_CURR and second column a categorical parameter
    returns:
        onehot_encoded: n onehot encoded columns for the n categories
        column_names: names of categories as a list
    """
    
    categories = data_in.unique()

    # get the values and replace nan with string N/A and replace numeric nan by aux value
    values = clean_nan(data_in)
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    if len(categories) != len(label_encoder.classes_):
        print('Warning '+index+' : num categories (', len(categories), ') doesn\'t match num encoder (', len(label_encoder.classes_), ')')
    
#     column_names = [str(x) + ' ('+index+')' for x in label_encoder.classes_]
    column_names = [str(x) for x in categories]
    return onehot_encoded, column_names

def binary_to_onehot(data_in):
    """
    takes in a pandas df with first column SK_ID_CURR and second column a categorical parameter
    returns:
        onehot_encoded: n onehot encoded columns for the n categories
        column_names: names of categories as a list
    """
    
    assert len(data_in.shape)==1 # we expect a pandas array with two columns
    
    categories = data_in.unique()
    
    assert len(categories)==2 # we expect two categories

    # get the values and replace nan with string N/A and replace numeric nan by aux value
    values = clean_nan(data_in)
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    if len(categories) != len(label_encoder.classes_):
        print('Warning '+index+' : num categories (', len(categories), ') doesn\'t match num encoder (', len(label_encoder.classes_), ')')
    
    onehot_encoded = np.array([onehot_encoded[:,0]]).T
    return onehot_encoded



def clean_numerical_data(data_in, verbose=False):
    """
     cleans up numerical data
    """
    
    imputer = Imputer()
    
    # replace nan values with mean
    data_out = imputer.fit_transform(data_in)
    
    scaler = StandardScaler()
    
    data_out = scaler.fit_transform(data_out)

    data_out = pd.DataFrame(data=data_out,    # values
     index=data_in.index,    # 1st column as index
     columns=data_in.keys())

    return data_out

def one_hot_encoding(data_in, max_categories=20, verbose=False):
    """
    one_hot_encoding data, e.g. no operations are done on the numerical values categorical and binary are encoded as one-hot
    
    returns: dataframe, where 
    
    """
    
    # descriptors_categorical: list of the columns names that are categorical
    # descriptors_binary: list of the columns names that are binary
    # descriptors_numeric: list of the columns names that are numeric
    descriptors_binary, descriptors_categorical, descriptors_numeric = get_descriptors(data_in,max_categories = max_categories, nan_treshold=0, verbose=verbose)

    data, column_names = [], []
    c = 0
    
    ##############################################################
    ### get all the categorical data and convert to onehot vectors
    ##############################################################
    print('get all the categorical data and convert to onehot vectors')
    sys.stdout.flush()
    for desc in tqdm(descriptors_categorical):
        
        onehot_encoded, names = categorical_to_onehot(data_in[desc])
        data.append(onehot_encoded)
        
        names = [x + ' ('+desc+')' for x in names]
        column_names += list(names)
    ##############################################################
    ### get all the binary data and convert to onehot vectors
    ##############################################################
    print('get all the binary data and convert to onehot vectors')
    sys.stdout.flush()
    for desc in tqdm(descriptors_binary):
        onehot_encoded = binary_to_onehot(data_in[desc])
        data.append(onehot_encoded)
        column_names += list([desc])
    
    data = np.concatenate(data, axis=1)
    # create dataframe
    X = pd.DataFrame(data=data,    # values
         index=data_in.index,    # 1st column as index
         columns=column_names)
    
    
    # add the numerical values to the dataframe
    data_numerical = data_in[descriptors_numeric]
    X = pd.concat([X, data_numerical],axis=1, verify_integrity=False)  # if slow set to verify_integrity=False
    
    return X

def preprocess_data_set(data_in, descriptors, dropna=False, verbose=False):
    """
    pre-process data, e.g. clean it so that we can use it for our model
    
    returns: feature vector X
    
    """

    data, column_names = [], []
    c = 0
    
    descriptors_categorical = descriptors['descriptors_categorical']  # list of the columns names that are categorical
    descriptors_binary = descriptors['descriptors_binary']  # list of the columns names that are binary
    descriptors_numeric = descriptors['descriptors_numeric']  # list of the columns names that are numeric
    
    ##############################################################
    ### get all the categorical data and convert to onehot vectors
    ##############################################################
    print('get the ', len(descriptors_categorical), ' categorical data and convert to onehot vectors')
    for desc in tqdm(descriptors_categorical):
        
        onehot_encoded, names = categorical_to_onehot(data_in[desc])
        data.append(onehot_encoded)
        
        names = [x + ' ('+desc+')' for x in names]
        column_names += list(names)
    ##############################################################
    ### get all the binary data and convert to onehot vectors
    ##############################################################
    print('get the ', len(descriptors_binary), ' binary data and convert to onehot vectors')
    for desc in tqdm(descriptors_binary):
        onehot_encoded = binary_to_onehot(data_in[desc])
        data.append(onehot_encoded)
        column_names += list([desc])
    
    data = np.concatenate(data, axis=1)
    # create dataframe
    X = pd.DataFrame(data=data,    # values
         index=data_in.index,    # 1st column as index
         columns=column_names)
    
    
#     # add the numerical values to the dataframe
    data_numerical = clean_numerical_data(data_in[descriptors_numeric], verbose=verbose)
    X = pd.concat([X, data_numerical],axis=1, verify_integrity=False)  # if slow set to verify_integrity=False

    if dropna:
        if verbose:
            print('number of data before drop na', len(X))

        # drop nan values
        X = X.dropna()

        if verbose:
            print('number of data after drop na', len(X))
    return X



