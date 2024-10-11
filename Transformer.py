import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import LayerNormalization, MultiHeadAttention
def loadFromPickle():
    with open("features_40", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels
def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels
def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Reshape((input_shape[0] * input_shape[1], 1))(x)  # Reshape input
    x = PositionalEncoding()(x)
    for _ in range(4):  # You can adjust the number of layers as needed
        x = TransformerEncoderBlock()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model
class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(start=0, limit=seq_length, delta=1)
        position = tf.cast(position, tf.float32)
        position *= 2 * np.pi / 40  # Assuming 40x40 image input
        position = tf.expand_dims(position, 0)
        position_encoding = tf.sin(position)
        position_encoding = tf.repeat(position_encoding, repeats=tf.shape(inputs)[-1], axis=-1)  # Repeat along the last dimension
        position_encoding = tf.expand_dims(position_encoding, 0)  # Add batch dimension
        position_encoding = tf.repeat(position_encoding, repeats=tf.shape(inputs)[0], axis=0)  # Repeat along batch dimension
        return inputs + position_encoding

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, ff_dim=64):
        super(TransformerEncoderBlock, self).__init__()
        self.ff_dim = ff_dim

    def build(self, input_shape):
        self.mha = MultiHeadAttention(num_heads=8, key_dim=64)
        self.ffn = tf.keras.Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.ff_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)  # Shape: [batch_size, seq_length, key_dim]
        attn_output = self.ffn(attn_output)  # Shape: [batch_size, seq_length, ff_dim]
        return attn_output
# Load data
features, labels = loadFromPickle()
features, labels = augmentData(features, labels)
features, labels = shuffle(features, labels)
# Split data into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=0)
# Build and compile the model
model = transformer_model(input_shape=(40, 40))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
# Train the model
model.fit(features_train, labels_train, validation_data=(features_test, labels_test), epochs=5, batch_size=64)
# Save the model
model.save('Autopilot_Transformer.keras')
