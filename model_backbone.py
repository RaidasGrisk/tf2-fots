import tensorflow as tf


class Backbone(tf.keras.Model):

    """
    Example of what is actually happening here.
    No loop for it makes it easier to understand.
    We are extracting 4 different layers from ResNet50/MobileNet.
    Dims will depend on the input size, but the relative
    sizes will stay the same and that is what matters.

    input       [1, 480, 640, 3]
    layer_1     [1, 15, 20, 2048]   (one of the last layers, 174th op in ResNet50)
    layer_2     [1, 30, 40, 1024]   (142)
    layer_3     [1, 60, 80, 512]    (80)
    layer_4     [1, 120, 160, 64]   (one of the first layers, 12th op in ResNet50)

    step 1: double the size of layer_1 -> [1, 30, 40, 2048]

    step 2: concat layer_1 [1, 30, 40, 2048] and layer_2 [1, 30, 40, 1024] -> [1, 30, 40, 3072]
            conv with stride 1 to -> [1, 30, 40, 128] this is just to decrease the num of filters (last dim)
            conv with stride 3 to -> [1, 30, 40, 128]
            double the size of this layer -> [1, 60, 80, 128]

    step 3: concat this [1, 60, 80, 128] and layer_2 [1, 60, 80, 512] -> [1, 60, 80, 640]
            conv to -> [1, 60, 80, 64]
            conv to -> [1, 60, 80, 64]
            resize  -> [1, 120, 160, 64]

    step 4: concat this [1, 120, 160, 64] and layer_4 [1, 120, 160, 64] -> [1, 120, 160, 128]
            conv to -> [1, 120, 160, 32]
            conv to -> [1, 120, 160, 32]
            conv to -> [1, 120, 160, 32] < -- these are the features we are going to be using in the model

    Last layer will serve us as the input for further branches of the model.

    """

    def __init__(self, backbone='resnet', input_shape=(480, 640, 3)):
        super(Backbone, self).__init__()

        self.backbone_name = backbone
        if backbone == 'mobilenet':
            self.baskbone = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape)
            self.layer_ids = [149, 69, 39, 24]
        else:
            self.baskbone = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
            self.layer_ids = [174, 142, 80, 12]

        self.baskbone.trainable = False
        self.backbone_layers = tf.keras.models.Model(
            inputs=self.baskbone.input,
            outputs=[self.baskbone.get_layer(index=i).output for i in self.layer_ids])

        self.l1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=tf.nn.relu)

        self.h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

        self.g1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

    def call(self, input):

        # layers extracted from Backbone model (ResNet or MobileNet):
        # 1st is the farthest one (near the end of the net),
        # 4th is the closest one (near the beggining)

        # preprocess input
        if 'mobile' in self.backbone_name.lower():
            input = tf.keras.applications.mobilenet.preprocess_input(input)
        else:
            input = tf.keras.applications.resnet50.preprocess_input(input)

        layer_1, layer_2, layer_3, layer_4 = self.backbone_layers(input)

        # step 1
        # layer_1 -> layer_1
        layer_shape = tf.shape(layer_1)
        layer_1 = tf.image.resize(layer_1, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 2
        # layer_1 + layer_2 -> layer_12
        layer_12_conc = self.l1(tf.concat([layer_1, layer_2], axis=-1))
        layer_12_conv = self.h1(layer_12_conc)
        layer_shape = tf.shape(layer_2)
        layer_12 = tf.image.resize(layer_12_conv, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 3
        # layer_12 + layer_3 -> layer_123
        layer_123_conc = self.l2(tf.concat([layer_12, layer_3], axis=-1))
        layer_123_conv = self.h2(layer_123_conc)
        layer_shape = tf.shape(layer_3)
        layer_123 = tf.image.resize(layer_123_conv, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 4
        # layer_123 + layer_4 -> layer_1234
        layer_1234_conc = self.l3(tf.concat([layer_123, layer_4], axis=-1))
        layer_1234_conv = self.h3(layer_1234_conc)
        layer_1234 = self.g1(layer_1234_conv)

        return layer_1234


# test
# model = Backbone()
# model.build(input_shape=[120, 120, 3])
# model.summary()
# for layer in model.layers:
#     print(layer.output_shape)
