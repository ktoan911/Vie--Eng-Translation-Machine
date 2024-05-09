import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    # print(q.shape, k.shape)
    QK = (1/tf.sqrt(tf.cast(tf.shape(k)
                            [-1], dtype=tf.float32))) * tf.matmul(q, k, transpose_b=True)
    # print("shape qkv", QKV.shape)
    if mask is not None:
        QK += (mask * -1e30)
    output = tf.matmul(tf.nn.softmax(QK, axis=-1), v)
    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        v = self.split_heads(self.wv(v), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        q = self.split_heads(self.wq(q), batch_size)

        output = scaled_dot_product_attention(q, k, v, mask)  # (1,8,60,16)
        output = tf.reshape(tf.transpose(
            output, perm=[0, 2, 1, 3]), (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output
