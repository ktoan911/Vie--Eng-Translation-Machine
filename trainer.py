import tensorflow as tf
import sacrebleu


class Trainer:
    def __init__(self, model, optimizer, epochs, model_path, start_token, end_token, tokenizer_en, tokenizer_vi):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.model_path = model_path
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer_en = tokenizer_en
        self.tokenizer_vi = tokenizer_vi

    def cal_acc(self, real, pred):
        pred = tf.argmax(pred, axis=2)
        real = tf.cast(real, pred.dtype)
        accuracies = tf.equal(real, pred)

        mask = tf.math.logical_not(real == 0)
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def compute_bleu(self, references, hypotheses):
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score

    def bleu_score(self, test_data):
        references = []
        hypotheses = []
        for batch in test_data:
            encoder_input, decoder_input = batch
            decoder_input_start = tf.convert_to_tensor(
                [[self.start_token]] * tf.shape(encoder_input['input_en'])[0].numpy(), dtype=tf.int64)
            # print(encoder_input['input_en'].shape, decoder_input.shape)
            predictions = self.predict(
                encoder_input['input_en'], decoder_input_start, is_train=False, max_length=decoder_input.shape[1])
            for ref, hyp in zip(self.detokenize_sentences(decoder_input.numpy()), predictions):
                # Convert to string and wrap in list
                references.append(ref)
                hypotheses.append(hyp)
        bleu_score = self.compute_bleu(references, hypotheses)
        return bleu_score

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

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            self.model.fit(train_data, validation_data=val_data, epochs=1)

        print('Saving checkpoint ......')
        self.model.save_weights(self.model_path)
        print('Saved checkpoint at {}'.format(self.model_path))
        print('----------------Done--------------------')

        print('Computing loss, accuracy, BLEU score in test data ......')
        test_loss, test_acc = self.model.evaluate(test_data)
        # Calculate BLEU score on the validation data
        print(
            f'Test Loss: {test_loss}, Test Accuracy: {test_acc}, BLEU Score: {self.bleu_score(test_data):.2f}')

    def build_model(self, input_shape):
        # Create dummy input to build the model
        dummy_input = {
            'input_en': tf.zeros((1, input_shape['input_en']), dtype=tf.int64),
            'input_vi': tf.zeros((1, input_shape['input_vi']), dtype=tf.int64)
        }
        self.model(dummy_input, training=False)

    def predict(self, encoder_input, decoder_input, is_train, max_length, max_repearations=2, max_token_near=2):

        def count_sublists(sublist, main_list, max_token_near):
            sublist_length = len(sublist)
            count = 0
            for i in range(len(main_list) - sublist_length + 1):
                if main_list[i:i + sublist_length] == sublist:
                    count += 1
                    # Nếu đã đếm được lớn hơn 2 lần, trả về false
                    if count > max_token_near:
                        return False
            return True

        self.build_model(
            {'input_en': encoder_input.shape[1], 'input_vi': decoder_input.shape[1]})
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function, metrics=[self.cal_acc])
        self.model.load_weights(self.model_path)
        print('-----------------------------------------')
        print('Predicting ......')

        results = []

        for encoder_input, decoder_input in zip(encoder_input, decoder_input):
            token_counts = {}
            past_token = self.start_token
            for _ in range(max_length):
                inputs = {
                    'input_en': tf.expand_dims(encoder_input, 0),
                    'input_vi': tf.expand_dims(decoder_input, 0),
                }

                preds = self.model(inputs, training=is_train)

                preds = preds[:, -1:, :]  # (batch_size, 1, vocab_size)

                predicted_id = tf.argmax(preds, axis=-1).numpy().item()

                # check số lần xuất hiện của từ
                if predicted_id in token_counts:
                    token_counts[predicted_id] += 1
                else:
                    token_counts[predicted_id] = 1

                if (token_counts[predicted_id] > max_repearations):
                    token_counts[predicted_id] -= 1
                    continue

                # check số lần 2 token cạnh nhau
                if not count_sublists([past_token, predicted_id], list(decoder_input), max_token_near):
                    continue

                decoder_input = tf.concat(
                    [decoder_input, [predicted_id]], axis=-1)
                past_token = predicted_id

                # return the result if the predicted_id is equal to the end token
                if predicted_id == self.end_token:
                    break
            results.append(decoder_input.numpy())

        return self.detokenize_sentences(results)

    def detokenize_sentences(self, tensor):
        detokenized_dataset = [self.tokenizer_vi.decode(ids) for ids in tensor]
        return [sentence.replace('_', '') for sentence in detokenized_dataset]
