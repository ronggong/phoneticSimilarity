from keras.models import Input
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense

def embedding_RNN(input_shape):

    input = Input(shape=input_shape)

    x = Bidirectional(LSTM(units=16, return_sequences=False))(input)

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=input, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    # model.summary()
    return model