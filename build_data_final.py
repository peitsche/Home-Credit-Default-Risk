import pandas as pd
import gc
import numpy as np
import utils
from sklearn.preprocessing import PolynomialFeatures, Imputer
# from sklearn.preprocessing import LabelEncoder, ,OneHotEncoder, StandardScaler
def aggregate_data(prev_subset, prefix, aggregates):
    """
    aggregates the data with identical SK_ID_CURR and creates new colunm for each value in aggregates
    
    
    """

    # aggregates the data 
    data_agg = prev_subset.groupby('SK_ID_CURR').agg(aggregates)
    # rename the colunmns
    if type(data_agg.columns) == pd.MultiIndex:
        # if multiindex we rename the column name by appending the levels
        data_agg.columns = pd.Index([prefix+'_' + e[0] + "_" + e[1].upper() for e in data_agg.columns.tolist()])
    else:
        # otherwise just add prefix
        data_agg.columns = pd.Index([prefix + '_' + e for e in data_agg.columns.tolist()])
    return data_agg


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    
    # read data
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    
    # NEW CLEANING
    #bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    #bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    #bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    bureau['DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_ENDDATE'].apply(lambda x: np.nan if x < -40000 else x)
    bureau['DAYS_CREDIT_UPDATE'] = bureau['DAYS_CREDIT_UPDATE'].apply(lambda x: np.nan if x < -40000 else x)
    bureau['DAYS_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'].apply(lambda x: np.nan if x < -40000 else x)

        
    # one hot encoding
    bb, bb_cat = utils.one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = utils.one_hot_encoder(bureau, nan_as_category)


    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(columns= 'SK_ID_BUREAU', inplace= True)
    del bb, bb_agg
    gc.collect()

    # print('asdasdad', bureau.columns)

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {cat:'mean' for cat in bureau_cat}
    cat_aggregations.update({cat + "_MEAN":'mean' for cat in bb_cat})
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(cat_aggregations)

    # bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])


    # Bureau: Total credits - using only numerical aggregations
    subset = aggregate_data(bureau, prefix='TOT', aggregates=num_aggregations)
    bureau_agg = bureau_agg.join(subset, how='left', on='SK_ID_CURR')

    # Bureau: Active credits - using only numerical aggregations
    subset = aggregate_data(bureau[bureau['CREDIT_ACTIVE_Active'] == 1],
                            prefix='ACT', aggregates=num_aggregations)
    bureau_agg = bureau_agg.join(subset, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    subset = aggregate_data(bureau[bureau['CREDIT_ACTIVE_Closed'] == 1],
                            prefix='CLS', aggregates=num_aggregations)
    bureau_agg = bureau_agg.join(subset, how='left', on='SK_ID_CURR')


    # # Bureau: Total credits - using only numerical aggregations
    # total_agg = bureau.groupby('SK_ID_CURR').agg(num_aggregations)
    # total_agg.columns = pd.Index(['TOT_' + e[0] + "_" + e[1].upper() for e in total_agg.columns.tolist()])
    # bureau_agg = bureau_agg.join(total_agg, how='left', on='SK_ID_CURR')
    # del total_agg
    # gc.collect()
    #
    # # Bureau: Active credits - using only numerical aggregations
    # active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    # active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    # active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    # bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    # del active, active_agg
    # gc.collect()
    #
    # # Bureau: Closed credits - using only numerical aggregations
    # closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    # closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    # closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    # bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    # del closed, closed_agg, bureau
    # gc.collect()
    #
    
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows=num_rows)
    prev, cat_cols = utils.one_hot_encoder(prev, nan_as_category=nan_as_category)
    # Days 365.243 values -> nan
    keys =['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    prev[keys] = prev[keys].replace(365243, np.nan)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']


    # Previous applications categorical features
    cat_aggregations = {cat : 'mean'for cat in cat_cols}
    prev_agg = aggregate_data(prev, 'PA', cat_aggregations)

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    # Previous Applications: Total Applications - only numerical features
    prev_subset = aggregate_data(prev,
                                 prefix='PREV', aggregates=num_aggregations)
    prev_agg = prev_agg.join(prev_subset, how='left', on='SK_ID_CURR')
    # Previous Applications: Approved Applications - only numerical features
    prev_subset = aggregate_data(prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1],
                                 prefix='APR', aggregates=num_aggregations)
    prev_agg = prev_agg.join(prev_subset, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    prev_subset = aggregate_data(prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1],
                                 prefix = 'REF', aggregates=num_aggregations)
    prev_agg = prev_agg.join(prev_subset, how='left', on='SK_ID_CURR')


        
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = utils.one_hot_encoder(pos, nan_as_category=nan_as_category)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = utils.one_hot_encoder(ins, nan_as_category= nan_as_category)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    """
    load the data from


    """

    cc = pd.read_csv('../input/credit_card_balance.csv', nrows=num_rows)

    # NEW CLEANING
    #cc['AMT_DRAWINGS_ATM_CURRENT'][cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    #cc['AMT_DRAWINGS_CURRENT'][cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
    cc['AMT_DRAWINGS_ATM_CURRENT'] = cc['AMT_DRAWINGS_ATM_CURRENT'].apply(lambda x: np.nan if x < 0 else x)
    cc['AMT_DRAWINGS_CURRENT'] = cc['AMT_DRAWINGS_CURRENT'].apply(lambda x: np.nan if x < 0 else x)
    
    cc, cat_cols = utils.one_hot_encoder(cc, nan_as_category=nan_as_category)
    # General aggregations
    cc.drop(columns=['SK_ID_PREV'], inplace=True)

    # aggregate the
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def add_features(df):


    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']


    feature_keys = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'PAYMENT_RATE', 'AMT_ANNUITY', 'DAYS_EMPLOYED']

    poly_features = df[feature_keys]

    imputer = Imputer(strategy = 'median')

    # replace nan values with median
    poly_features = imputer.fit_transform(poly_features)

    poly_transformer = PolynomialFeatures(degree=3)
    poly_transformer.fit(poly_features)
    poly_features = poly_transformer.transform(poly_features)

    poly_features = pd.DataFrame(data=poly_features,columns=poly_transformer.get_feature_names(feature_keys),
                               index= df.index)

    poly_features.drop(feature_keys,axis=1, inplace=True)



    df = df.join(poly_features, how='left', on='SK_ID_CURR')
    return df


def build_data(num_rows=None, max_categories=5, verbose=True):



    # load main data
    data_application_train = pd.read_csv('../input/application_train.csv', nrows=num_rows).set_index('SK_ID_CURR')
    data_application_test = pd.read_csv('../input/application_test.csv', nrows=num_rows).set_index('SK_ID_CURR')
    data_application_full = data_application_train.append(data_application_test, sort=True)
    if verbose:
        print('loaded application_train', data_application_full.shape)

    # clean up some of the data
    data_application_full['FONDKAPREMONT_MODE'] = data_application_full['FONDKAPREMONT_MODE'].replace('not specified', np.nan)
    data_application_full['DAYS_EMPLOYED'] = data_application_full['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # NEW CLEANING    
    data_application_full['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # one hot encode the main data
    df, new_cat = utils.one_hot_encoder(data_application_full, nan_as_category=True)

    del data_application_train, data_application_test
    gc.collect()

    if verbose:
        print('one hot encoded main data', df.shape)


    # load other data
    bb = bureau_and_balance(num_rows=num_rows)
    pa = previous_applications(num_rows=num_rows)
    ps = pos_cash(num_rows=num_rows)
    ip = installments_payments(num_rows=num_rows)
    cc = credit_card_balance(num_rows=num_rows)


    print('bureau_and_balance', bb.shape)
    print('previous_applications', pa.shape)
    print('pos_cash', ps.shape)
    print('installments_payments', ip.shape)
    print('credit_card_balance', cc.shape)

    # combine into single dataset
    df = df.join(bb, how='left', on='SK_ID_CURR')
    if verbose:
        print('added bureau_and_balance', df.shape)
    df = df.join(pa, how='left', on='SK_ID_CURR')
    if verbose:
        print('added previous_applications', df.shape)
    df = df.join(ps, how='left', on='SK_ID_CURR')
    if verbose:
        print('added pos_cash', df.shape)
    df = df.join(ip, how='left', on='SK_ID_CURR')
    if verbose:
        print('added installments_payments', df.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    if verbose:
        print('added credit_card_balance', df.shape)



    # 365243 should be nan
    # df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df.replace(np.inf, np.nan, inplace=True)


    df = add_features(df)
    if verbose:
        print('added new features', df.shape)

    # drop columns where all the values are identical
    utils.drop_single_values(df)
    if verbose:
        print('cleaned columns with single values', df.shape)


    return df