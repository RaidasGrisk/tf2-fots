import tensorflow as tf


class Recognition(tf.keras.Model):

    """
    Shape of recognizer features (2, 8, 64, 32)
    +Shape of recognizer features (2, 4, 64, 256) max_pool [2, 2], [2, 1]
    ++Shape of recognizer features (2, 2, 64, 256) max_pool [2, 2], [2, 1]
    +++Shape of recognizer features (2, 1, 64, 256) max_pool [2, 2], [2, 1]
    ++++Shape of recognizer features (2, 1, 64, 128) conv
    ++++Shape of word_vec (64, 2, 128) reshape
    ++++Shape of logits (2, 64, 46) lstm + fully_connected [b, times, NUM_CLASSES]
    """

    def __init__(self, num_classes, training=True, drop_prob=0.0):
        super(Recognition, self).__init__()
        self.drop_prob = drop_prob


        # cnn
        # 1st block
        self.layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization(trainable=training, momentum=0.997, epsilon=0.00001)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='same')

        # 2nd block
        self.layer_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization(trainable=training, momentum=0.997, epsilon=0.00001)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='same')

        # 3rd block
        self.layer_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn_3 = tf.keras.layers.BatchNormalization(trainable=training, momentum=0.997, epsilon=0.00001)
        self.pool_3 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='same')

        # # rnn
        lstm_fw_cell_1 = tf.keras.layers.LSTM(128, return_sequences=True, unit_forget_bias=True, dropout=self.drop_prob)
        lstm_bw_cell_1 = tf.keras.layers.LSTM(128, go_backwards=True, return_sequences=True, unit_forget_bias=True, dropout=self.drop_prob)
        self.bilstm_1 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_1, backward_layer=lstm_bw_cell_1)

        lstm_fw_cell_2 = tf.keras.layers.LSTM(128, return_sequences=True, unit_forget_bias=True, dropout=self.drop_prob)
        lstm_bw_cell_2 = tf.keras.layers.LSTM(128, go_backwards=True, unit_forget_bias=True, return_sequences=True, dropout=self.drop_prob)
        self.bilstm_2 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_2, backward_layer=lstm_bw_cell_2)

        # From the paper: To avoid overfitting on small training datasets
        # like ICDAR 2015, we add dropout before fully-connection.
        # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Dropout
        self.dense = tf.keras.layers.Dense(num_classes)  # number of classes + 1 blank char

    def __call__(self, input):

        # 1st block
        x = self.layer_1(input)
        x = self.bn_1(x)
        x = tf.nn.relu(x)  # activation after bn
        x = self.pool_1(x)

        # 2nd block
        x = self.layer_2(x)
        x = self.bn_2(x)
        x = tf.nn.relu(x)
        x = self.pool_2(x)

        # 3nd block
        x = self.layer_3(x)
        x = self.bn_3(x)
        x = tf.nn.relu(x)
        x = self.pool_3(x)

        # rnn
        x = tf.squeeze(x, axis=[1])  # [BATCH, TIME, FILTERS] because height of tensor is now 1
        x = self.bilstm_1(x)
        x = self.bilstm_2(x)
        # print([i.sum() for i in x.numpy()])

        logits = self.dense(x)

        return logits

    @staticmethod
    def loss_recognition(y, logits, ws):

        indices, values, dense_shape = y
        y_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        label_length = tf.math.bincount(indices[:, 0])
        loss = tf.nn.ctc_loss(labels=y_sparse,
                              logits=tf.transpose(logits, [1, 0, 2]),
                              label_length=label_length,
                              logit_length=[len(logits[-1]) for _ in logits],  # logits [batch, time, nclass]
                              blank_index=-1)  # -1 will reproduce the behavior of using num_classes-1 for the blank
        return tf.reduce_mean(loss)

# # ------- #
# # import numpy as np
# #
# # input = (np.random.randint(1, 10, size=(7, 64, 256)) / 10).astype(np.float32)
# # tf.keras.layers.LSTM(126, return_sequences=True)(input).shape
# # tf.keras.layers.LSTM(126, go_backwards=True, return_sequences=True)(input).shape
# # tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(126, return_sequences=True),
# #                               backward_layer=tf.keras.layers.LSTM(126, go_backwards=True, return_sequences=True))(input).shape
