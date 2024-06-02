import tensorflow as tf
import Layers.position_encoding as pe
import Layers.multihead_attention as mha
import Layers.encoder as enc


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
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
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, d_model, input_length=maximum_position_encoding)
        self.pos_encoding = pe.positional_encoding(
            maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_decoder_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x = dec_layer(x, enc_output, training=training, padding_mask=padding_mask, look_ahead_mask=look_ahead_mask)
            # print(f"After DecoderLayer {i+1}:", x.shape)

        return x
