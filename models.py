import keras.models as kmodels
import keras.layers as klayers

def LSTMOnly(dimFeature):
    
    # LSTM model
    model = kmodels.Sequential()
    model.add(klayers.LSTM(100, input_shape=(1, dimFeature), dropout=0.2, recurrent_dropout=0.2))
    model.add(klayers.Dense(100, activation='relu'))
    model.add(klayers.Dense(1, activation='sigmoid'))

    return model

def convNetLSTM(dimFeature):
    
    # Convnet model
    filters = 16
    kernel_size = 16
    """"
    model = kmodels.Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        klayers.Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(a, b)),
        klayers.MaxPooling1D(),     # Downsample the output of convolution by 2X.
        #klayers.Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        #klayers.MaxPooling1D(),
        klayers.Flatten(),
        klayers.Dense(1, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
    ))
    """
    model = kmodels.Sequential()
    #model.add(klayers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(1, trainFeatures.shape[2]), data_format="channels_first"))
    model.add(klayers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(1, dimFeature), data_format="channels_first"))
    model.add(klayers.MaxPool1D())
    #model.add(klayers.LSTM(10, dropout=0.25, recurrent_dropout=0.25))
    model.add(klayers.LSTM(50, dropout=0.25, recurrent_dropout=0.25))
    model.add(klayers.Dense(16, activation="relu"))
    model.add(klayers.Dense(1, activation="sigmoid"))
    
    return model

def RNNOnly(dimFeature):
    
    # RNN model
    model = kmodels.Sequential()
    model.add(klayers.SimpleRNN(50, input_shape=(1, dimFeature), dropout=0.2, recurrent_dropout=0.2))
    model.add(klayers.Dense(50, activation='relu'))
    model.add(klayers.Dense(1, activation='sigmoid'))

    return model

def convNetRNN(dimFeature):
    
    # Convnet model
    filters = 16
    kernel_size = 16
    # RNN model
    model = kmodels.Sequential()
    model.add(klayers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(1, dimFeature), data_format="channels_first"))
    model.add(klayers.MaxPool1D())
    model.add(klayers.SimpleRNN(50, input_shape=(1, dimFeature), dropout=0.2, recurrent_dropout=0.2))
    model.add(klayers.Dense(50, activation='relu'))
    model.add(klayers.Dense(1, activation='sigmoid'))

    return model


