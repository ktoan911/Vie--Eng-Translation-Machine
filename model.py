import tensorflow as tf
import Layers.decoder as decoder
import Layers.encoder as encoder
import Layers.generator_mask as gm


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, enc_padding_mask, look_ahead_mask, dec_padding_mask, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = encoder.EncoderPack(
            num_layers, dff, num_heads, input_vocab_size, pe_input, rate)
        self.decoder = decoder.DecoderPack(
            num_layers, dff, num_heads, target_vocab_size, pe_target, rate)
        self.enc_padding_mask = enc_padding_mask
        self.look_ahead_mask = look_ahead_mask
        self.dec_padding_mask = dec_padding_mask

    def call(self, x, training):
        inp, tar = x
        enc_output = self.encoder(inp, training, self.enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, self.look_ahead_mask, self.dec_padding_mask)
        return dec_output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
