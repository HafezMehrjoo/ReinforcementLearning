import keras.layers as layer
import keras.models as keras_model


def deep_learning_model(X, y):
    model = keras_model.Sequential()
    model.add(layer.Dense(units=32, input_dim=4))
    model.add(layer.Dense(units=128, activation='relu'))
    model.add(layer.Dense(units=2, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    return model
