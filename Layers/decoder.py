import tensorflow as tf
import Layers.position_encoding as pe
import Layers.multihead_attention as mha
import Layers.encoder as enc


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_mha = mha.MultiHeadAttention(d_model, num_heads)
        self.pad_mha = mha.MultiHeadAttention(d_model, num_heads)
        self.ffn = enc.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training=True, padding_mask=None, look_ahead_mask=None):
        inp = self.masked_mha(x, x, x, mask=look_ahead_mask)
        out1 = self.dropout1(inp, training=training)
        q = self.layernorm1(out1 + x)

        k = v = enc_output
        out2 = self.pad_mha(q, k, v, mask=padding_mask)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out2 + q)

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

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for _ in range(self.num_decoder_layers):
            x = self.dec_layer(x, enc_output, training=training,
                               padding_mask=padding_mask, look_ahead_mask=look_ahead_mask)

        return x


def test():
    d_model = 512
    num_heads = 8
    dff = 2048
    target_vocab_size = 8000
    maximum_position_encoding = 5000
    num_decoder_layers = 2
    rate = 0.1
    decoder = DecoderPack(num_decoder_layers, d_model, num_heads, dff,
                          target_vocab_size, maximum_position_encoding, rate)
    x = tf.random.uniform((64, 60), dtype=tf.int64, minval=0, maxval=200)
    enc_output = tf.random.uniform((64, 60, 512))
    look_ahead_mask = tf.random.uniform((64, 1, 1, 60))
    padding_mask = tf.random.uniform((64, 1, 1, 60))
    out = decoder(x, enc_output, training=True,
                  look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    print(out.shape)
