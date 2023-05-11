import numpy as np
import pandas as pd
import random
import time
import csv
import os
import itertools
import ta
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from mylibs.training_part import delete_small_change, class_weight, create_model, generate_sequence

import tushare as ts
pro = ts.pro_api('c12abd8696482b6d604d47293faef7efe9fbfe14db6b18b1669eeab1')


random_seed = 1


def process_data_jingyi_shen(data):
    """calculate technical indicators and label for one stock"""
    data['change'] = data.close - data.close.shift(1)
    data['ret'] = data.close.pct_change()
    data['sma_10'] = data.close.rolling(10).mean()

    macd = ta.trend.MACD(data.close, 26, 12, 8)
    data['macd'] = macd.macd()
    data['macd_hist'] = macd.macd_diff()
    data['macd_signal'] = macd.macd_signal()

    data['mtm_10'] = data.close/data.close.shift(10) * 100
    data['roc_10'] = (data.close - data.close.shift(10))/data.close.shift(10) * 100

    rsi = ta.momentum.RSIIndicator(data.close, 5)
    data['rsi_5'] = rsi.rsi()

    wnr = ta.momentum.WilliamsRIndicator(data.high, data.low, data.close, 9)
    data['wnr_9'] = wnr.williams_r()

    slow = ta.momentum.StochRSIIndicator(data.close, 14, 3, 3)
    data['slowk'] = slow.stochrsi_k()
    data['slowd'] = slow.stochrsi_d()

    data['adosc'] = (data.high - data.close.shift())/(data.high - data.low + 1e-4)
    data['ar_26'] = (data.high.rolling(26).sum() - data.open.rolling(26).sum()) / \
                    (data.open.rolling(26).sum() - data.low.rolling(26).sum() + 1e-4)
    data['br_26'] = (data.high.rolling(26).sum() - data.open.shift().rolling(26).sum()) / \
                    (data.open.shift().rolling(26).sum() - data.low.rolling(26).sum() + 1e-4)
    data['bias_20'] = (data.close - data.close.rolling(20).mean()) / data.close.rolling(20).mean()
    data['label'] = np.where(data.close.shift(-1) > data.close, 1, 0)
    '''no label for last day'''
    data = data[:-1]
    return data


def get_data_jingyi_shen(stock_list, start_date='20140101', end_date='20211231'):
    df = pd.DataFrame()
    '''put data for all stocks in one dataframe'''
    for i in stock_list:
        '''
        ts_code: symbol for Chinese market stock, something like 600519.SH
        trade_date: trading date
        open_qfq: qfq means adjusted (for dividend, split, etc.)
        vol: volume
        cci: Commodity Channel Index
        '''
        fields = 'ts_code,trade_date,open_qfq,high_qfq,low_qfq,close_qfq,vol,amount,cci'
        df_aux = pro.stk_factor(ts_code=i, start_date=start_date, end_date=end_date, fields=fields)
        '''rename columns in convenient ones'''
        df_aux = df_aux.rename(columns={'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close'})
        '''!!!sort by date!!!'''
        df_aux['trade_date'] = pd.to_datetime(df_aux['trade_date'])
        df_aux = df_aux.sort_values(by='trade_date')

        df_aux = process_data_jingyi_shen(df_aux)

        df = pd.concat([df, df_aux])
    return df


def feature_extension(data, stock_list):
    """
    according to original paper, extend features by
    min max scaling,
    polarization: 1 for bigger than 0, 0 for smaller or equal to 0,
    and fluctuation percentage: percentage change,
    polarize and fluctuation do not suffer from data leakage, special process is needed for min max scaling
    """
    data = data.dropna(axis=0)
    new_data = pd.DataFrame()
    polarize_list = ['macd', 'macd_signal', 'macd_hist', 'cci', 'mtm_10', 'roc_10', 'wnr_9', 'adosc', 'bias_20']
    max_min_list = ['vol', 'amount', 'sma_10', 'rsi_5', 'wnr_9', 'slowk', 'slowd', 'adosc', 'ar_26', 'br_26']
    fluctuation_list = ['sma_10', 'mtm_10', 'roc_10', 'rsi_5', 'slowk', 'slowd']
    for i in stock_list:
        stock_df = data[data['ts_code'] == i].copy()
        for j in polarize_list:
            stock_df[f'{j}_plr'] = np.where(stock_df[j] > 0, 1, 0)
        for j in max_min_list:
            scaler = MinMaxScaler()
            aux = np.array(stock_df[j]).reshape(-1, 1)
            '''only fit the scaler with the first 80% of data to avoid data leakage'''
            scaler.fit(aux[:int(len(stock_df)*0.8)])
            stock_df[f'{j}_maxmin'] = scaler.transform(aux)
        for j in fluctuation_list:
            stock_df[f'{j}_flc'] = (stock_df[j] - stock_df[j].shift(1)) / (stock_df[j].shift(1) + 1e-4)
        new_data = pd.concat([new_data, stock_df])
    new_data = new_data.dropna(axis=0)
    return new_data


'''----------Recursive feature elimination----------'''


def rfe_selection(X_train, y_train, n_features_to_select):
    """use RFE to select n features based on one stock"""
    classifier = LogisticRegression(max_iter=300000, random_state=random_seed)
    selector = RFE(classifier, n_features_to_select=n_features_to_select, step=1)
    selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.support_].tolist()
    return selected_features


def rfe_selection_with_all_stocks(stock_list, data, n_features_to_select):
    """implement RFE on all stocks and choose the features that are selected by all RFE selectors on different stocks"""
    selected_feature_dict = {}
    for i in stock_list:
        target_data = data[data['ts_code'] == i].set_index('trade_date')
        target_data = target_data.drop(['ts_code', 'open', 'high', 'low', 'close'], axis=1).sort_index().copy()
        '''only use training dataset to select features'''
        X_train, X_test, y_train, y_test = train_test_split(target_data.drop('label', axis=1), target_data['label'],
                                                            test_size=0.2, shuffle=False)
        selected_features = rfe_selection(X_train, y_train, n_features_to_select)
        selected_feature_dict[i] = selected_features

    '''put all selected features in one list then use collections.Counter to count the number'''
    big_package = []
    total_n = len(selected_feature_dict.values())
    for i in selected_feature_dict.values():
        big_package += i

    '''if the number equals the number of stocks, then it means one feature is selected by all selectors'''
    final_selected_features = []
    for i, j in Counter(big_package).items():
        if j == total_n:
            final_selected_features.append(i)
    print(f'{len(final_selected_features)} features are selected')
    return final_selected_features


'''----------Principle component analysis----------'''


def do_pca(n, X_train, X_test, verbose=True):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    if verbose:
        print(pca.explained_variance_ratio_.sum())
    return X_train, X_test


'''----------Grid search----------'''


def grid_search_shen(df, grid_search_path, split_span_0, validation_split_span_1,
                     lookback_span_2, batch_size_span_3, hu_span_4, learning_rate_span_5,
                     start_year_span_6, layers_span_7, label_threshold_span_8, activation_function
                     ):
    my_callbacks = [EarlyStopping(patience=30, monitor='val_loss', mode='auto', verbose=0, restore_best_weights=True)]

    '''restore the hyperparameters, metrics and other detailed information in a csv file'''
    with open(grid_search_path, 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(
            ['max_train_accuracy', 'max_val_accuracy', 'test_accuracy', 'min_train_loss', 'min_val_loss',
             'test_loss', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1',
             'split', 'validation_split', 'lookback', 'batch_size', 'hu_span', 'learning_rate',
             'layers', 'label_threshold', 'start_year', 'time', 'probability'
             ])

        '''use itertools.product() to write the loop'''
        combos = itertools.product(split_span_0, validation_split_span_1, lookback_span_2,
                                   batch_size_span_3, hu_span_4,
                                   learning_rate_span_5, start_year_span_6, layers_span_7,
                                   label_threshold_span_8)
        '''to log the process'''
        length = len([i for i in combos])
        j = 1

        '''values in iterator can only be red once, so do that again'''
        combos = itertools.product(split_span_0, validation_split_span_1, lookback_span_2,
                                   batch_size_span_3, hu_span_4,
                                   learning_rate_span_5, start_year_span_6, layers_span_7,
                                   label_threshold_span_8)

        for i in combos:
            '''calculate time for one loop'''
            start_time = time.time()

            '''read the values in iterator'''
            split = i[0]
            validation_split = i[1]
            lookback = i[2]
            batch_size = i[3]
            hu = i[4]
            learning_rate = i[5]
            start_year = str(i[6])
            layers = i[7]
            label_threshold = i[8]

            '''divide training and testing data set'''
            X_train, X_test, y_train, y_test = train_test_split(
                                                df[start_year:'2022'].drop(['ret', 'label'], axis=1),
                                                df[start_year:'2022'][['ret', 'label']].values,
                                                test_size=split, shuffle=False)
            X_train, X_test = do_pca(20, X_train, X_test, False)
            '''generate sequence data'''
            X_train, y_train = generate_sequence(X_train, y_train, sequence_length=lookback)
            X_test, y_test = generate_sequence(X_test, y_test, sequence_length=lookback)
            '''delete samples with return within certain small range'''
            X_train, y_train = delete_small_change(X_train, y_train, label_threshold)
            '''I use 'ret' y_train/y_test for delete_small_change function, so I need to remove it'''
            y_train = np.delete(y_train, 0, axis=1).flatten().astype(int)
            y_test = np.delete(y_test, 0, axis=1).flatten().astype(int)
            '''create model'''
            model = create_model(X_train, hu=hu, layers=layers,
                                 learning_rate=learning_rate, activation_function=activation_function)
            history = model.fit(X_train, y_train, epochs=1000, validation_split=validation_split,
                                validation_freq=1,
                                class_weight=class_weight(X_train, y_train), callbacks=my_callbacks,
                                batch_size=batch_size, verbose=0, workers=6, shuffle=False)
            '''calculate the metrics I want to record'''
            res = pd.DataFrame(history.history)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_prob = model.predict(X_test, verbose=0)
            y_pred = [1 if i >= 0.5 else 0 for i in y_prob]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            precision = tp / (tp + fp + 1e-4)
            recall = tp / (tp + fn + 1e-4)
            f1 = 2 * precision * recall / (precision + recall + 1e-4)
            '''calculate time for one loop'''
            one_loop_time = (time.time() - start_time) / 60
            '''to log the process'''
            print(f'{j}/{length}', f'  {one_loop_time} minutes')
            j += 1

            '''one row for csv'''
            current_combination = pd.DataFrame({
                'max_train_accuracy': [max(res['accuracy'])],
                'max_val_accuracy': [max(res['val_accuracy'])],
                'test_accuracy': [accuracy],
                'min_train_loss': [min(res['loss'])],
                'min_val_loss': [min(res['val_loss'])],
                'test_loss': [loss],
                'tp': [tp],
                'fp': [fp],
                'tn': [tn],
                'fn': [fn],
                'precision': [precision],
                'recall': [recall],
                'f1': [f1],
                'split': [split],
                'validation_split': [validation_split],
                'lookback': [lookback],
                'batch_size': [batch_size],
                'hu_span': [hu],
                'learning_rate': [learning_rate],
                'layers': [layers],
                'label_threshold': [label_threshold],
                'start_year': [start_year],
                'time': [one_loop_time],
                'probability': [y_prob],
            })
            csv_write.writerow(list(current_combination.values.flatten()))
