import tensorflow as tf
from keras.layers import Embedding, Dense, Embedding, Dropout, LayerNormalization, GlobalAveragePooling1D
from Transformer_Encoder import positioncal_encoding as pe
from Transformer_Encoder import multihead_attention as mha
from keras import Model


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = mha.MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def __call__(self, x, training, mask):
        inp1 = self.mha(x, x, x, mask=mask)
        out1 = self.dropout1(inp1)
        out1 = self.layernorm1(out1 + x)

        inp2 = self.ffn(inp1)
        out2 = self.dropout2(inp2)
        out2 = self.layernorm2(out2 + out1)
        return out2


class TransformerEncoderPack(Model):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoderPack, self).__init__()
        # Khai báo các lớp cần sử dụng
        self.embedding = Embedding(
            input_vocab_size, d_model)  # input_length=maximum_position_encoding
        self.pos_encoding = pe.positional_encoding(
            maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(
            d_model, num_heads, dff, rate) for i in range(num_encoder_layers)]
        self.dropout = Dropout(rate)
        self.globalaveragepooling1d = GlobalAveragePooling1D()
        self.dense = Dense(1, activation='sigmoid')

    def __call__(self, x, training=True):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=None)
        x = self.globalaveragepooling1d(x)
        x = self.dense(x)
        return x
