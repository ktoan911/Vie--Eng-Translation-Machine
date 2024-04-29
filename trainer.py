
import tensorflow as tf
from Layers.generator_mask import generate_mask


class Trainer:
    def __init__(self, model, optimizer, epochs, checkpoint_folder):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_folder, max_to_keep=3)

    def cal_acc(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(real == 0)
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def loss_function(self, real, pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = cross_entropy(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def fit(self, train_dataset, val_dataset):

        # tar_inp = tar[:, :-1]
        # tar_real = tar[:, 1:]
        # encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(
        #     inp, tar_inp)
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function, metrics=[self.cal_acc])

        self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset,
                       callbacks=[self.checkpoint_manager])
        print('----------------Done--------------------')

    def predict(self, encoder_input, decoder_input, is_train, max_length, end_token):
        print('=============Inference Progress================')
        print('----------------Begin--------------------')
        # Loading checkpoint
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')

        for i in range(max_length):

            encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(
                encoder_input, decoder_input)

            preds = self.model(encoder_input, decoder_input, is_train,
                               encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
            # print('---> preds', preds)

            preds = preds[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(preds, axis=-1)

            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                break

        return decoder_input
