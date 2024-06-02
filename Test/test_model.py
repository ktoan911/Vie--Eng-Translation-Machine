import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
import model



def test_model():
    d_model = 512
    num_heads = 8
    num_layers = 6
    dff = 2048
    input_vocab_size = 8500
    target_vocab_size = 8000
    pe_input = 10000
    pe_target = 6000

    # Instantiate the Transformer model
    transformer = model.Transformer(num_layers, d_model, num_heads, dff,
                                    input_vocab_size, target_vocab_size, pe_input, pe_target)

    # Create input tensors
    input_en = tf.random.uniform((64, 38))
    input_de = tf.random.uniform((64, 37))

    # Create a dictionary with the inputs
    inputs = {
        'input_en': input_en,
        'input_vi': input_de,
    }

    # Pass the inputs dictionary to the transformer
    result = transformer(inputs=inputs, training=True)

    print(result)


if __name__ == "__main__":
    test_model()
