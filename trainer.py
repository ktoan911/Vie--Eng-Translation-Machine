from Layers.generator_mask import generate_mask
import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, epochs, checkpoint_folder):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.checkpoint_path = checkpoint_folder

    def cal_acc(self, real, pred):
        pred = tf.argmax(pred, axis=2)
        real = tf.cast(real, pred.dtype)
        accuracies = tf.equal(real, pred)

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

    def fit(self, train_data, val_data, test_data):
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function, metrics=[self.cal_acc])
        self.model.fit(train_data, validation_data=val_data,
                       epochs=self.epochs)

        print('Saving checkpoint ......')
        self.model.save(self.checkpoint_path, save_format="tf")
        print('Saved checkpoint at {}'.format(self.checkpoint_path))
        print('----------------Done--------------------')
        test_loss, test_acc = self.model.evaluate(test_data)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

    def evaluate(self, test_data, model):
        test_loss, test_acc = model.evaluate(test_data)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

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
