import tensorflow as tf
import positioncal_encoding as pe
import multihead_attention as mha
import encoder as enc


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = mha.MultiHeadAttention(d_model, num_heads)
        self.mha2 = mha.MultiHeadAttention(d_model, num_heads)
        self.ffn = enc.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training=True, padding_mask=None, look_ahead_mask=None):
        inp = self.mha1(x, x, x, mask=look_ahead_mask)
        out1 = self.dropout1(inp, training=training)
        out1 = self.layernorm1(out1 + x)

        q = k = enc_output
        out2 = self.mha2(q, k, out1, mask=padding_mask)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out2 + out1)

        out3 = self.ffn(out2)
        out3 = self.dropout3(out3, training=training)
        out3 = self.layernorm3(out3 + out2)
        return out3


class DecoderPack(tf.keras.layers.Layer):
    def __init__(self, num_decoder_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(DecoderPack, self).__init__()
        self.num_decoder_layers = num_decoder_layers
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, d_model, input_length=maximum_position_encoding)
        self.pos_encoding = pe.positional_encoding(
            maximum_position_encoding, d_model)
        self.dec_layer = DecoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.dense(
            target_vocab_size, activation="softmax")

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for _ in range(self.num_decoder_layers):
            x = self.dec_layer(x, enc_output, training,
                               padding_mask, look_ahead_mask)
        output = self.final_layer(x)

        return output
