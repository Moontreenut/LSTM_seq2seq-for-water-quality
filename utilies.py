import IPython
import math
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import shutil
from keras_tuner import HyperModel, RandomSearch

def data_load(data_path, item, look_back, look_forward):
    raw_data = pd.read_csv(data_path)
    indi_data = raw_data[[item]]
    helper_data = []
    wq_data = []
    for i in range(indi_data.shape[0] - look_forward - look_back):
        piece = indi_data.iloc[i: i + look_back + look_forward, :]
        if piece.isna().sum().sum() == 0:
            helper_data.append(piece.values)
    helper_data = np.array(helper_data)
    if helper_data.shape != (0,):
        wq_data.append(helper_data)
    wq_data = np.vstack(wq_data)

    data_mean = np.mean(wq_data)
    data_std = np.std(wq_data)
    norm_data = ((wq_data - data_mean) / data_std)
    length = norm_data.shape[0]

    boundary_1 = int(0.6*length)
    boundary_2 = int(0.8*length)
    np.random.shuffle(norm_data)

    train_data = norm_data[:boundary_1, :, :]
    valid_data = norm_data[boundary_1:boundary_2, :, :]
    test_data = norm_data[boundary_2:, :, :]
    return train_data, valid_data, test_data, data_mean, data_std

def data_split_norm(train_data, valid_data, test_data, look_back, look_forward):
    train_x = train_data[:,:look_back,:]
    train_y = train_data[:,look_back:look_back+look_forward,]

    valid_x = valid_data[:,:look_back,:]
    valid_y = valid_data[:,look_back:look_back+look_forward,]

    test_x = test_data[:,:look_back,:]
    test_y = test_data[:,look_back:look_back+look_forward,]

    train_decoder = np.zeros((train_y.shape[0], train_y.shape[1], 1))
    valid_decoder = np.zeros((valid_y.shape[0], valid_y.shape[1], 1))
    test_decoder = np.zeros((test_y.shape[0], test_y.shape[1], 1))

    return (train_x, train_decoder), (valid_x, valid_decoder), (test_x, test_decoder), train_y, valid_y, test_y

def NSE(y_true, y_pred):
    variance_square = np.square(y_true - np.mean(y_true))
    variance_square_sum = np.sum(variance_square)
    error_square_sum = np.sum(np.square(y_pred - y_true))
    nse = 1 - np.divide(error_square_sum, variance_square_sum)
    return nse

def NRMSE(y_true, y_pred):
    mse = np.sum(np.square(y_pred - y_true))/len(y_true)
    rmse = np.sqrt(mse)
    nrmse = rmse/(np.mean(y_true))
    return nrmse

def RMSE(y_true, y_pred):
    mse = np.sum(np.square(y_pred - y_true))/len(y_true)
    rmse = np.sqrt(mse)
    return rmse

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

def fvu_error(y_true, y_pred):
    y = tf.math.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    x = tf.math.reduce_sum(tf.square(y_pred - y_true))
    return tf.divide(x, y)

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        encoder_hidden_sizes = [hp.Int('hidden_size_' + str(1), min_value=2, max_value=20, step=2), hp.Int('hidden_size_' + str(2), min_value=2, max_value=20, step=2)]
        encoder_inputs = keras.Input(shape=self.input_shape)
        encoder_cells = []

        for hidden_size in encoder_hidden_sizes:
            encoder_cells.append(keras.layers.LSTMCell(hidden_size,
                                                      activation='tanh',
                                                      ))
        encoder = keras.layers.RNN(encoder_cells, return_state=True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        encoder_states = encoder_outputs_and_states[1:]

        decoder_inputs = keras.Input(shape=(None, 1))
        decoder_cells = []
        for hidden_size in encoder_hidden_sizes:
            decoder_cells.append(keras.layers.LSTMCell(hidden_size,
                                                       activation='tanh',
                                                       ))
        decoder = keras.layers.RNN(decoder_cells, return_state=True, return_sequences=True)
        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

        decoder_outputs = decoder_outputs_and_states[0]
        decoder_dense = keras.layers.Dense(1, activation='tanh')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      loss=fvu_error,
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsolutePercentageError(),
                               ]
                      )
        return model

def hyper_tuner(log_path, time_step, train_data, valid_data, test_data, item, mode, look_forward):
    train_x, valid_x, test_x, train_y, valid_y, test_y = data_split_norm(train_data, valid_data, test_data, time_step, look_forward)

    hyper_model = MyHyperModel(input_shape=train_x[0].shape[-2:])
    project_name = item + "_" + str(time_step)
    tuner = RandomSearch(
        hyper_model,
        objective='val_loss',
        max_trials=25,
        executions_per_trial=5,
        directory=log_path,
        project_name=project_name)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001),
        ClearTrainingOutput()
    ]
    tuner.search_space_summary()
    tuner.search(train_x, train_y, epochs=300, batch_size=64,
                 validation_data=(valid_x, valid_y),
                 callbacks=callbacks,
                 verbose=2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model_info = {"item": item,
                  "time_step": time_step,
                  "look_forward": look_forward,
                  "hidden_size_1": best_hps.get('hidden_size_1'),
                  "hidden_size_2": best_hps.get('hidden_size_2'),
                  }

    del tuner

    return model_info

def build_model(input_shape, hidden_size_list):
    encoder_hidden_sizes = hidden_size_list
    encoder_inputs = keras.Input(shape=input_shape)
    encoder_cells = []

    for hidden_size in encoder_hidden_sizes:
        encoder_cells.append(keras.layers.LSTMCell(hidden_size,
                                                   activation='tanh',
                                                   ))
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

    decoder_inputs = keras.Input(shape=(None, 1))
    decoder_cells = []
    for hidden_size in encoder_hidden_sizes:
        decoder_cells.append(keras.layers.LSTMCell(hidden_size,
                                                   activation='tanh',
                                                   ))
    decoder = keras.layers.RNN(decoder_cells, return_state=True, return_sequences=True)
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]
    decoder_dense = keras.layers.Dense(1, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss=fvu_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsolutePercentageError(),
                           ]
                  )
    return model

def training(train_data, valid_data, test_data, item, time_step, hidden_size_list, main_path, save_name, mode, look_forward,data_mean, data_std):
    train_x, valid_x, test_x, train_y, valid_y, test_y = data_split_norm(train_data, valid_data, test_data, time_step, look_forward)
    model_result = pd.DataFrame()
    j = -1
    for repeat in range(1, 16):
        print("*"*50, "repeat=", repeat, "*"*50)
        j = j+1
        keras.backend.clear_session()
        model = build_model(input_shape=train_x[0].shape[-2:], hidden_size_list=hidden_size_list)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001),
        ]
        history = model.fit(train_x, train_y, epochs=300, batch_size=64,
                            validation_data=(valid_x, valid_y),
                            callbacks=callbacks,
                            verbose=2)
        model_name = item + "_" + str(repeat) + "_model.tf"
        model_path = os.path.join(main_path, model_name)
        tf.saved_model.save(model, model_path)

        train_fvu, train_rmse, train_mre = history.history['loss'][-1], history.history['root_mean_squared_error'][-1], history.history["mean_absolute_percentage_error"][-1]

        test_fvu, test_rmse, test_mre = model.evaluate(test_x, test_y, verbose=0)

        predict = model.predict(test_x) 
        predict = predict.reshape((predict.shape[0], predict.shape[1]))

        test_ture = pd.DataFrame(test_y.reshape(test_y.shape[0], test_y.shape[1]))
        
        test_ture_my = test_ture * data_std + data_mean
        predict_my = predict * data_std + data_mean

        test_nse_my = NSE(test_ture_my, predict_my)
        test_rmse_my = RMSE(test_ture_my, predict_my)
        test_nrmse_my = NRMSE(test_ture_my, predict_my)

        model_info = {'item': item,
                      "time_steps": str(time_step),
                      'repeat': str(repeat),
                      'train_fvu': train_fvu,
                      'train_rmse': train_rmse,
                      'train_mre': train_mre,
                      "test_fvu": test_fvu,
                      'test_rmse': test_rmse_my,
                      'test_nrmse': test_nse_my,
                      'test_nse': test_nrmse_my,
                      }
        model_info = pd.DataFrame(model_info, index=[j])
        model_result = pd.concat([model_result, model_info])

    results_path = os.path.join(main_path, save_name)
    if not os.path.exists(results_path):
        model_result.to_csv(results_path, encoding='utf-8', index=None)
    else:
        model_result.to_csv(results_path, mode='a', header=False, index=None)

   
