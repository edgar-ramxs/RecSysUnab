import tensorflow as tf

class TargetModel:
    def __init__(self, input_shape, learning_rate=0.001, dropout_rate=0.2, layer_units=[64, 32]):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(layer_units[0], activation='relu', input_shape=(input_shape,)))
        self.model.add(tf.keras.layers.Dropout(dropout_rate))
        
        for units in layer_units[1:]:
            self.model.add(tf.keras.layers.Dense(units, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(dropout_rate))
        
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    def train(self, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return history
    
    def evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Loss en el conjunto de prueba: {loss}')
        return loss
    
    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions


