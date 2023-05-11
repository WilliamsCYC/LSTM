import pandas as pd
import numpy as np

import datetime
import random
import time
import csv
import os

import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, LSTM, Dense, Dropout, InputLayer, Conv1D, Conv2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
plt.style.use('seaborn')


def show_lib_version():
    """print the version of python libraries"""
    import sklearn
    try:
        print('My tensorflow version: ', tf.__version__)
        print('Is tensorflow-gpu available: ', tf.test.is_gpu_available())
    except:
        print('No tensorflow')
    print('My sklearn version: ', sklearn.__version__)
    print('My numpy version: ', np.__version__)
    print('My pandas version: ', pd.__version__)


def retrieve_data(ts_code, file_name):
    """
    I want to test different threshold to delete small return samples
    so all the data fed int other model should contain return
    but there are no 'ret' column in other csv except for basic_data.csv
    file_name can be:
    basic_data
    prediction_data_k_means_random_feature
    prediction_data_k_pca
    signals
    """
    basic_data = pd.read_csv(f'./data/{ts_code}/basic_data.csv', index_col='trade_date', parse_dates=True)
    basic_data['label'] = np.where(basic_data['close'].shift(-1) > basic_data['close'] * 1.00025, 1, 0)
    '''the last day has no label'''
    basic_data = basic_data[:-1]
    if file_name == 'basic_data':
        df = basic_data
    else:
        df = pd.read_csv(f'./data/{ts_code}/{file_name}.csv', index_col='trade_date', parse_dates=True)
        df = pd.merge(df, basic_data[['label', 'ret']], how='left', on='trade_date')
    df.dropna(axis=0, inplace=True)
    print(f'{df.shape[0]} rows, {df.shape[1]} columns')

    if file_name in ['prediction_data_k_pca', 'signals']:
        scale = False
    else:
        scale = True
    no_scale_list = None
    return df, scale, no_scale_list


def scale_features(X_train, X_test, no_scale_list, scale):
    """no_scale_list: not really has any function in my submission"""
    '''scale: for some dataset the features do not need to be scaled'''
    if scale:
        scaler = MinMaxScaler()
        if no_scale_list is None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train_unscaled = X_train[no_scale_list].values
            X_test_unscaled = X_test[no_scale_list].values
            if len(X_train.drop(no_scale_list, axis=1)) != 0:
                X_train_scaled = scaler.fit_transform(X_train.drop(no_scale_list, axis=1))
                X_test_scaled = scaler.transform(X_test.drop(no_scale_list, axis=1))
                X_train = np.concatenate((X_train_scaled, X_train_unscaled), axis=1)
                X_test = np.concatenate((X_test_scaled, X_test_unscaled), axis=1)
            else:
                '''in case some data have no feature that should be scaled'''
                X_train = X_train_unscaled
                X_test = X_test_unscaled
    else:
        X_train = X_train.values
        X_test = X_test.values
    return X_train, X_test


def generate_sequence(X, y, sequence_length):
    """manipulate the data os that each time step is a sample and each sample has the features of past lookback days"""
    X_train = []
    y_train = []
    '''len(X)+1 to get the whole range(save one sample)'''
    for i in range(sequence_length, len(X)+1):
        X_train.append(X[i - sequence_length:i, :])
        '''!!!my label is attached to the last day of lookback period, so -1 in y[i-1] is very important!!!'''
        y_train.append(y[i-1])
    return np.array(X_train), np.array(y_train)


def class_weight(X, y):
    """balance the label"""
    c0, c1 = np.bincount(y)
    w0 = (1 / c0) * (len(X)) / 2
    w1 = (1 / c1) * (len(X)) / 2
    return {0: w0, 1: w1}


def create_model(X_train, hu, layers, learning_rate, l2_lambda=0.001,
                 activation_function='relu'):
    """make learning rate decay with time"""
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=1,
        staircase=False)
    '''use Adam optimizer'''
    optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-08)

    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))

    for i in range(1, layers + 1):
        '''use hyperparameter layers to control the number of LSTM layer'''
        if i == layers:
            model.add(LSTM(units=hu, return_sequences=False, kernel_regularizer=l2(l2_lambda),
                           name=f'LSTM{i}'))
        else:
            model.add(LSTM(units=hu, return_sequences=True, kernel_regularizer=l2(l2_lambda),
                           name=f'LSTM{i}'))
        model.add(Activation(activation_function))
        model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_lambda), name='Output'),)
    model.compile(optimizer=optimizer, loss='mse', metrics='accuracy')
    return model


def plot_process(history):
    """plot the process of loss and accuracy"""
    res = pd.DataFrame(history.history)

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(res['accuracy'], label='Training Accuracy')
    plt.plot(res['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(res['loss'], label='Training Loss')
    plt.plot(res['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def evaluate(X_test, y_test, model):
    """
    evaluate the result and plot confusion matrix and ROC curve
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('-'*60, 'test data', '-'*60)

    'generate y_pred label according to the probability given by model'
    y_pred = model.predict(X_test, verbose=0)
    y_pred = [1 if i >= 0.5 else 0 for i in y_pred]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-4)
    recall = tp / (tp + fn + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)

    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.imshow([[tp, fn], [fp, tn]], interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], [1, 0])
    plt.yticks([0, 1], [1, 0])
    plt.text(0, 0, str(tp))
    plt.text(0, 1, str(fp))
    plt.text(1, 0, str(fn))
    plt.text(1, 1, str(tn))
    plt.title("Confusion matrix")
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC curve (area = {roc_auc.round(2)})")
    plt.show()

    print('loss: ', loss)
    print('accuracy: ', accuracy)
    print('TP: ', tp, '   FP: ', fp, '   TN: ', tn, '   FN: ', fn)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)


def print_basic_info(X_train, X_test, model):
    """monitor the number of sample"""
    print('-'*60, 'basic info', '-'*60)
    print(f'number of training sample: {len(X_train)}')
    print(f'number of testing sample: {len(X_test)}')
    print('-'*60, 'model summary', '-'*60)
    print(model.summary())


def delete_small_change(X_train, y_train, label_threshold):
    """delete samples with return within certain small range"""
    redundant_sample_list = []
    for i in range(y_train.shape[0]):
        if label_threshold >= y_train[i][0] >= -label_threshold:
            redundant_sample_list.append(i)
    X_train = np.delete(X_train, redundant_sample_list, axis=0)
    y_train = np.delete(y_train, redundant_sample_list, axis=0)
    return X_train, y_train


def my_callback(file_name):
    """early stop"""
    early_stop = EarlyStopping(patience=30, monitor='val_loss', mode='auto', verbose=0, restore_best_weights=True)

    '''tensorboard log path'''
    log_dir = f'logs/{file_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    """a callback to show the progress of training, one epoch one dot, one line for 100 epochs"""
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
            print('.', end='')

    my_callbacks = [early_stop, tensorboard_callback, PrintDot()]
    return my_callbacks


'''----------integrate every thing for one run----------'''


def one_run_inner(X_train, X_test, y_train, y_test, file_name, label_threshold, hu, layers,
                  learning_rate, activation_function, batch_size, validation_split):
    """
    create model, train model, evaluate model and return probability
    X_train, X_test, y_train, y_test should be well-prepared in a sequential structure
    """
    X_train, y_train = delete_small_change(X_train, y_train, label_threshold)

    '''I use 'ret' y_train/y_test for delete_small_change function, so I need to remove it '''
    y_train = np.delete(y_train, 0, axis=1).flatten().astype(int)
    y_test = np.delete(y_test, 0, axis=1).flatten().astype(int)

    model = create_model(X_train, hu=hu, layers=layers, learning_rate=learning_rate,
                         activation_function=activation_function)

    print_basic_info(X_train, X_test, model)

    my_callbacks = my_callback(file_name)

    """train the model"""
    history = model.fit(X_train, y_train, epochs=1000, validation_split=validation_split, validation_freq=1,
                        class_weight=class_weight(X_train, y_train), callbacks=my_callbacks, batch_size=batch_size,
                        verbose=0, workers=6, shuffle=False)

    '''get history and output metrics'''
    plot_process(history)
    evaluate(X_test, y_test, model)
    y_pred = model.predict(X_test, verbose=0)
    '''return the probability for backtesting'''
    return y_pred


def one_run_outer(df, no_scale_list, file_name, split, validation_split, lookback, hu, batch_size, learning_rate,
                  ts_code, layers, label_threshold, activation_function, scale):
    """split data set, scale data, generate sequence then run one_run_inner"""
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['ret', 'label'], axis=1),
                                                        df[['ret', 'label']].values, test_size=split, shuffle=False)
    '''prepare data for backtesting'''
    back_test_data = X_test[lookback - 1:].copy()

    X_train, X_test = scale_features(X_train, X_test, no_scale_list, scale)

    '''generate sequence data'''
    X_train, y_train = generate_sequence(X_train, y_train, sequence_length=lookback)
    X_test, y_test = generate_sequence(X_test, y_test, sequence_length=lookback)

    y_pred = one_run_inner(X_train, X_test, y_train, y_test, file_name, hu=hu, layers=layers,
                           label_threshold=label_threshold, learning_rate=learning_rate,
                           activation_function=activation_function,
                           batch_size=batch_size, validation_split=validation_split)

    '''prepare data for backtesting'''
    back_test_data['probability'] = y_pred
    back_test_data = back_test_data.reset_index()[['trade_date', 'probability']]
    back_test_data.to_csv(f'./results/{ts_code}/back_test_{file_name}.csv', sep=',', index=False)


'''----------Grid search----------'''


def grid_search(df, no_scale_list, grid_search_path, split_span_0, validation_split_span_1,
                lookback_span_2, batch_size_span_3, hu_span_4, learning_rate_span_5,
                start_year_span_6, layers_span_7, label_threshold_span_8, scale, activation_function
                ):
    my_callbacks = [EarlyStopping(patience=30, monitor='val_loss', mode='auto', verbose=0, restore_best_weights=True)]

    '''store the hyperparameters, metrics and other detailed information in a csv file'''
    with open(grid_search_path, 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(
            ['max_train_accuracy', 'max_val_accuracy', 'test_accuracy', 'min_train_loss', 'min_val_loss', 'test_loss',
             'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1',
             'split', 'validation_split', 'lookback', 'batch_size', 'hu_span', 'learning_rate',
             'layers', 'label_threshold', 'start_year', 'time', 'probability'
             ])
        '''use itertools.product() to write the loop'''
        combos = itertools.product(split_span_0, validation_split_span_1, lookback_span_2, batch_size_span_3, hu_span_4,
                                   learning_rate_span_5, start_year_span_6, layers_span_7, label_threshold_span_8)
        '''to log the process'''
        length = len([i for i in combos])
        j = 1
        '''values in iterator can only be red once, so do that again'''
        combos = itertools.product(split_span_0, validation_split_span_1, lookback_span_2, batch_size_span_3, hu_span_4,
                                   learning_rate_span_5, start_year_span_6, layers_span_7, label_threshold_span_8)

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
            X_train, X_test, y_train, y_test = train_test_split(df[start_year:'2022'].drop(['ret', 'label'], axis=1),
                                                                df[start_year:'2022'][['ret', 'label']].values,
                                                                test_size=split, shuffle=False)
            X_train, X_test = scale_features(X_train, X_test, no_scale_list, scale)
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
            '''train model'''
            history = model.fit(X_train, y_train, epochs=1000, validation_split=validation_split, validation_freq=1,
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
            print(f'{j}/{length}', f'  {round(one_loop_time, 2)} minutes')
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


def choose_best_hyper(ts_code, file_name):
    """choose best hyperparameter"""
    if ts_code is None:
        grid_search_path = './results/jingyi_shen/grid_search_result.csv'
    else:
        grid_search_path = f'./results/{ts_code}/grid_search_{file_name}.csv'
    grid_search_result = pd.read_csv(grid_search_path)
    '''ignore if over 80% of prediction falls in one direction'''
    grid_search_result['p_over_n'] = (grid_search_result['tp'] + grid_search_result['fp']) / (grid_search_result['tn'] + grid_search_result['fn'])
    grid_search_result = grid_search_result[(grid_search_result['p_over_n'] >= 0.25) & (grid_search_result['p_over_n'] <= 4)]
    '''
    choose results with validation accuracy higher than 60%
    then choose the results with top test accuracy
    '''
    first_30 = grid_search_result[grid_search_result['max_val_accuracy'] > 0.6].sort_values(by='test_accuracy').head(1)
    feature_list = ['split', 'validation_split', 'lookback', 'batch_size', 'hu_span', 'learning_rate', 'layers',
                    'label_threshold', 'start_year']
    feature_df = first_30[feature_list].T
    feature_df.rename(columns={int(f'{feature_df.columns[0]}'): 'value'}, inplace=True)
    return feature_df
