import tensorflow as tf
import Layers.decoder as decoder
import Layers.encoder as encoder
import Layers.generator_mask as gm


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


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = encoder.EncoderPack(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = decoder.DecoderPack(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(
            target_vocab_size)

    def call(self, inputs, training=True):
        inp, out = inputs
        encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = gm.generate_mask(
            inp, out)
        enc_output = self.encoder(
            inp, training=training, mask=encoder_padding_mask)
        dec_output = self.decoder(
            out, enc_output, training=training, look_ahead_mask=decoder_look_ahead_mask, padding_mask=decoder_padding_mask)
        output = self.final_layer(dec_output)
        return output


# def test():
#     d_model = 512
#     num_heads = 8
#     num_layers = 6
#     dff = 2048
#     input_vocab_size = 8500
#     target_vocab_size = 8000
#     pe_input = 10000
#     pe_target = 6000
#     transformer = Transformer(num_layers, d_model, num_heads, dff,
#                               input_vocab_size, target_vocab_size, pe_input, pe_target)
#     result = transformer(inputs=(tf.random.uniform(
#         (64, 38)), tf.random.uniform((64, 37))))

#     print(result.shape)


# test()
