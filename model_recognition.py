import tensorflow as tf

#
# class Recognition(tf.keras.Model):
#
#     """
#     Shape of recognizer features (2, 8, 64, 32)
#     +Shape of recognizer features (2, 4, 64, 256) max_pool [2, 2], [2, 1]
#     ++Shape of recognizer features (2, 2, 64, 256) max_pool [2, 2], [2, 1]
#     +++Shape of recognizer features (2, 1, 64, 256) max_pool [2, 2], [2, 1]
#     ++++Shape of recognizer features (2, 1, 64, 128) conv
#     ++++Shape of word_vec (64, 2, 128) reshape
#     ++++Shape of logits (2, 64, 46) lstm + fully_connected [b, times, NUM_CLASSES]
#     """
#
#     def __init__(self, num_classes):
#         super(Recognition, self).__init__()
#
#         # cnn
#         self.layer_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1], padding='same')
#
#         self.layer_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_8 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1], padding='same')
#
#         self.layer_9 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_10 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_11 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
#         self.layer_12 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1], padding='same')
#
#         # rnn
#         lstm_fw_cell_1 = tf.keras.layers.LSTM(128, return_sequences=True)
#         lstm_bw_cell_1 = tf.keras.layers.LSTM(128, go_backwards=True, return_sequences=True)
#         self.birnn1 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_1, backward_layer=lstm_bw_cell_1)
#
#         lstm_fw_cell_2 = tf.keras.layers.LSTM(128, return_sequences=True)
#         lstm_bw_cell_2 = tf.keras.layers.LSTM(128, go_backwards=True, return_sequences=True)
#         self.birnn2 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_2, backward_layer=lstm_bw_cell_2)
#
#         self.dense = tf.keras.layers.Dense(num_classes)  # number of classes + 1 blank char
#
#     def __call__(self, input):
#
#         # cnn
#         x = self.layer_1(input)
#         x = self.layer_2(x)
#         x = self.layer_3(x)
#         x = self.layer_4(x)
#
#         x = self.layer_5(x)
#         x = self.layer_6(x)
#         x = self.layer_7(x)
#         x = self.layer_8(x)
#
#         x = self.layer_9(x)
#         x = self.layer_10(x)
#         x = self.layer_11(x)
#         x = self.layer_12(x)
#
#         # rnn
#         x = tf.squeeze(x, axis=[1])  # [BATCH, TIME, FILTERS] because height of tensor is now 1
#         x = self.birnn1(x)
#         x = self.birnn2(x)
#
#         logits = self.dense(x)
#
#         return logits
#
#     @staticmethod
#     def loss_recognition(y, logits, ws):
#         indices, values, dense_shape = y
#         y_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
#         label_length = tf.math.bincount(indices[:, 0])
#         loss = tf.nn.ctc_loss(labels=y_sparse,
#                               logits=tf.transpose(logits, [1, 0, 2]),
#                               label_length=label_length,
#                               logit_length=[64 for i in logits],  # len(logits)
#                               blank_index=63)   # len(logits)+1 ?
#         return tf.reduce_mean(loss)


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

    def __init__(self, num_classes, training=True):
        super(Recognition, self).__init__()

        # cnn
        self.layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, dtype='float32')
        self.layer_2 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1])

        self.layer_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.layer_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1])

        self.layer_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")
        self.layer_6 = tf.keras.layers.BatchNormalization(trainable=training)

        self.layer_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.layer_8 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1])

        # rnn
        lstm_fw_cell_1 = tf.keras.layers.LSTM(126, return_sequences=True)
        lstm_bw_cell_1 = tf.keras.layers.LSTM(126, go_backwards=True, return_sequences=True)
        self.birnn1 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_1, backward_layer=lstm_bw_cell_1)

        lstm_fw_cell_2 = tf.keras.layers.LSTM(126, return_sequences=True)
        lstm_bw_cell_2 = tf.keras.layers.LSTM(126, go_backwards=True, return_sequences=True)
        self.birnn2 = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_2, backward_layer=lstm_bw_cell_2)

        self.dense = tf.keras.layers.Dense(num_classes)  # number of classes + 1 blank char

    def __call__(self, input):

        # cnn
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.layer_5(x)
        x = self.layer_6(x)
        x = tf.nn.relu(x)
        x = self.layer_7(x)
        x = self.layer_8(x)

        # rnn
        x = tf.squeeze(x, axis=[1])  # [BATCH, TIME, FILTERS] because height of tensor is now 1
        x = self.birnn1(x)
        x = self.birnn2(x)

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
                              logit_length=[64 for i in logits],  # len(logits)
                              blank_index=63)   # len(logits)+1 ?
        return tf.reduce_mean(loss)
