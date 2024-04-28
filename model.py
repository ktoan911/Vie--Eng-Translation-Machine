import tensorflow as tf
import Layers.decoder as decoder
import Layers.encoder as encoder
import Layers.generator_mask as gm


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = encoder.TransformerEncoderPack(
            num_layers, dff, num_heads, input_vocab_size, pe_input, rate)
        self.decoder = decoder.TransformerDecoderPack(
            num_layers, dff, num_heads, target_vocab_size, pe_target, rate)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        return dec_output
