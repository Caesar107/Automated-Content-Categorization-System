# 人脸实战——步骤2：提取特征

## 1 – 网络模型构建

### 根据网络模型构建mobilenetv2网络


```python
def Mobilefacenet_Arcface(input_shape=None, inplanes=64,
                  classes=1000,
                  num_feature=128,
                  setting=Mobilefacenet_bottleneck_setting):
    input = Input(shape=input_shape)
    y = Input(shape=(classes,))
    x = conv_block(input, d_in=3, d_out=64, kernel_size=3, stride=2, padding=1)
    x = conv_block(x, d_in=64, d_out=64, kernel_size=3, stride=1, padding=1, depthwise=True)
    for t, c, n, s in setting:
        for i in range(n):
            if i == 0:
                x = bottleneck(x, inplanes, c, s, t)
            else:
                x = bottleneck(x, inplanes, c, 1, t)
            inplanes = c
    x = conv_block(x, d_in=128, d_out=512, kernel_size=1, stride=1, padding=0)
    x = conv_block(x, d_in=512, d_out=512, kernel_size=(7, 6), stride=1, padding=0,
                   depthwise=True, linear=True)
    x = conv_block(x, d_in=512, d_out=num_feature, kernel_size=1, stride=1, padding=0,
                   linear=True)
    x = keras.layers.Flatten()(x)
    y = Input(shape=(classes,))
    output = ArcFace_v2(n_classes=classes)((x, y))
    return Model([input, y], output)


if __name__ == '__main__':
    model = Mobilefacenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=10572)
    model.summary()
```




定义了一个名为Mobilefacenet_Arcface的函数，该函数构建了一个具有特定结构的人脸识别模型。

函数的输入参数如下：

- input_shape：输入张量的形状。默认为None，表示可以接受任意形状的输入。
- inplanes：初始通道数，默认为64。
- classes：类别数，即人脸识别任务中的身份数量，默认为1000。
- num_feature：特征向量的维度，默认为128。
- setting：Mobilefacenet_bottleneck_setting，一个列表，定义了模型中每个bottleneck模块的参数设置。

开始通过定义一个输入张量和一个用于标签的张量。然后通过调用conv_block函数创建了一系列卷积块来构建模型的主体部分。

在卷积块的构建过程中，使用了bottleneck函数来定义bottleneck模块。bottleneck模块是一种特殊的卷积块，包含了深度可分离卷积、批归一化和激活函数等操作。该函数根据给定的参数设置循环构建bottleneck模块。

之后，通过两个卷积块对特征进行进一步处理。最后，通过一个Flatten层将特征展平为一维向量。

然后定义了一个ArcFace_v2层，它是用于人脸识别任务的一种损失函数层，可以增强特征向量之间的差异。ArcFace_v2层的输入是特征向量和标签。将特征向量和标签作为输入传递给ArcFace_v2层，并将输出作为模型的输出。

最后，在主程序中调用Mobilefacenet_Arcface函数来创建一个模型，并打印出模型的摘要信息。模型的输入形状为(112, 96, 3)，特征向量维度为128，类别数为10572。模型的摘要信息可以显示模型的层结构和参数数量等信息。

### 构建arcface函数


```python
import math

import tensorflow as tf

# arc face loss calculation
class ArcFace(tf.keras.layers.Layer):

    def __init__(self, n_classes=10, s=32.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        x, y = inputs
        c = tf.shape(x)[-1]
        # normalize feature
        x = tf.math.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.math.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = {"n_classes": self.n_classes,
                  "s": self.s,
                  "m":self.m
                  }
        base_config = super(ArcFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ArcFace_v2(tf.keras.layers.Layer):
    '''
    Arguments:
        inputs: the input embedding vectors
        n_classes: number of classes
        s: scaler value (default as 64)
        m: the margin value (default as 0.5)
    Returns:
        the final calculated outputs
    '''

    def __init__(self, n_classes, s=32., m=0.5, **kwargs):
        self.init = tf.keras.initializers.get('glorot_uniform')  # Xavier uniform intializer
        self.n_classes = n_classes
        self.s = s
        self.m = m
        super(ArcFace_v2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer=self.init)
        super(ArcFace_v2, self).build(input_shape[0])

    def call(self, inputs, **kwargs):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)

        # features
        #X = inputs[0]
        # 1-D or one-hot label works as mask
        #Y_mask = inputs[1]
        X, Y_mask = inputs
        # If Y_mask is not in one-hot form, transfer it to one-hot form.
        #if tf.shape(Y_mask)[-1] == 1:
        #    Y_mask = tf.cast(Y_mask, tf.int32)
        #    Y_mask = tf.reshape(tf.one_hot(Y_mask, self.n_classes), (-1, self.n_classes))

        X_normed = tf.math.l2_normalize(X, axis=1)  # L2 Normalized X
        W = tf.math.l2_normalize(self.W, axis=0)  # L2 Normalized Weights

        # cos(theta + m)
        cos_theta = tf.keras.backend.dot(X_normed, W)  # ����˷�
        cos_theta2 = tf.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = tf.sqrt(sin_theta2 + tf.keras.backend.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))

        # This condition controls the theta + m should in range [0, pi]
        #   0 <= theta + m < = pi
        #   -m <= theta <= pi - m
        cond_v = cos_theta - threshold
        cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)

        # mask by label
        # Y_mask =+ K.epsilon()
        inv_mask = 1. - Y_mask
        s_cos_theta = self.s * cos_theta

        output = tf.nn.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * Y_mask))

        return output

    def get_config(self):
        config = {"n_classes": self.n_classes,
                  "s": self.s,
                  "m":self.m
                  }
        base_config = super(ArcFace_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_classes

```

定义两个自定义的Keras层：ArcFace和ArcFace_v2。这些层用于计算ArcFace损失函数，用于人脸识别任务中的特征向量的训练和预测。

ArcFace层：
- 初始化方法（__init__）：接受以下参数：n_classes（类别数）、s（缩放因子，默认为32.0）、m（margin值，默认为0.50）和regularizer（正则化器，默认为None）。
- build方法：根据输入张量的形状构建层，并创建可训练的权重W。
- call方法：接收输入x和标签y作为输入。首先对输入的特征向量x进行L2归一化，然后对权重W进行L2归一化。接下来，计算特征向量x和权重W之间的内积得到logits。然后，根据ArcFace损失函数的定义，通过计算acos和加上margin来调整logits。最后，将调整后的logits进行缩放和softmax操作，并返回输出结果。
- compute_output_shape方法：计算输出的形状。
- get_config方法：返回层的配置信息。

ArcFace_v2层：
- 初始化方法（__init__）：接受n_classes（类别数）、s（缩放因子，默认为32.0）和m（margin值，默认为0.5）作为参数。
- build方法：根据输入张量的形状构建层，并创建可训练的权重W。
- call方法：接收输入x和标签的掩码Y_mask作为输入。首先对输入的特征向量x进行L2归一化，然后对权重W进行L2归一化。接下来，根据ArcFace_v2损失函数的定义，计算cos(theta + m)和sin(theta)的值。然后，根据条件和公式计算调整后的cos(theta + m)的值，并将其进行softmax操作。最后，返回输出结果。
- compute_output_shape方法：计算输出的形状。
- get_config方法：返回层的配置信息。

这两个层可以用于构建人脸识别模型，结合卷积网络和这些自定义层进行训练和预测。

1. `ArcFace`层：

- `__init__`方法：初始化函数，接收参数`n_classes`（类别数）、`s`（缩放因子，默认为32.0）、`m`（margin值，默认为0.50）和`regularizer`（正则化器，默认为None）。
- `build`方法：构建函数，根据输入张量的形状建立层，并创建可训练的权重`W`。
- `call`方法：调用函数，接收输入`inputs`（包含特征向量`x`和标签`y`）作为参数。首先对特征向量`x`进行L2归一化，然后对权重`W`进行L2归一化。计算特征向量和权重之间的点积`logits`。然后根据ArcFace损失函数的定义，通过计算acos和加上margin来调整`logits`。最后将调整后的`logits`进行缩放和softmax操作，并返回输出结果。
- `compute_output_shape`方法：计算输出的形状。
- `get_config`方法：返回层的配置信息。

2. `ArcFace_v2`层：

- `__init__`方法：初始化函数，接收参数`n_classes`（类别数）、`s`（缩放因子，默认为32.0）和`m`（margin值，默认为0.5）。
- `build`方法：构建函数，根据输入张量的形状建立层，并创建可训练的权重`W`。
- `call`方法：调用函数，接收输入`inputs`（包含特征向量`x`和标签的掩码`Y_mask`）作为参数。首先对特征向量`x`进行L2归一化，然后对权重`W`进行L2归一化。根据ArcFace_v2损失函数的定义，计算`cos(theta + m)`和`sin(theta)`的值。根据条件和公式计算调整后的`cos(theta + m)`的值，并进行softmax操作。最后返回输出结果。
- `compute_output_shape`方法：计算输出的形状。
- `get_config`方法：返回层的配置信息。

这两个层分别用于计算ArcFace损失函数的两个版本，可以在人脸识别模型中用于训练和预测过程中的特征向量处理和损失计算。

## 2 – 模型训练

### 1)数据处理
将CASIA-WebFace数据集分成训练数据集和测试数据集。
读取CASIA-WebFace-112X96.txt，使用train_test_split方法将数据分成训练数据集和测试数据集


```python
# 数据路径
data_root = "data/CASIA"
img_txt_dir = os.path.join(data_root, 'CASIA-WebFace-112X96.txt')


def load_dataset(val_split=0.05):
    image_list = []     # image directory
    label_list = []     # label
    with open(img_txt_dir) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        image_dir, label_name = info.split(' ')
        image_list.append(os.path.join(data_root, 'CASIA-WebFace-112X96', image_dir))
        label_list.append(int(label_name))

    trainX, testX, trainy, testy = train_test_split(image_list, label_list, test_size=val_split)

    return trainX, testX, trainy, testy
```





定义`load_dataset`函数，用于加载数据集并将其拆分为训练集和测试集。

函数说明如下：

- `val_split`参数：用于指定测试集的比例，默认为0.05，即将5%的数据作为测试集。

函数流程：

1. 定义空的`image_list`和`label_list`列表，用于存储图像路径和对应的标签。
2. 使用`open`函数打开`img_txt_dir`路径下的文件，该文件包含图像路径和标签信息。
3. 使用`read`方法读取文件内容，并使用`splitlines`方法将其按行分割，得到图像路径和标签的列表`img_label_list`。
4. 对于`img_label_list`中的每个元素（表示一个图像的路径和标签），通过`split`方法将其分割为图像路径和标签名，并分别添加到`image_list`和`label_list`中。
5. 使用`train_test_split`函数将`image_list`和`label_list`按照指定的`val_split`比例进行拆分，得到训练集的图像路径、测试集的图像路径、训练集的标签和测试集的标签。
6. 返回训练集图像路径、测试集图像路径、训练集标签和测试集标签。


```python

```

使用Dataset.from_tensor_slices把数据进行切片处理；将数据映射到preprocess方法中，对image以及label进行预处理以及batch size处理


```python
def preprocess(x,y):
    # x: directory，y：label
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [112, 96])

    x = tf.image.random_flip_left_right(x)

    # x: [0,255]=> -1~1
    x = (tf.cast(x, dtype=tf.float32) - 127.5) / 128.0
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=class_num)

    return (x, y), y

# get data slices
train_image, val_image, train_label, val_lable = load_dataset()

# get class number
class_num = len(np.unique(train_label))

batchsize = 64
db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))     # construct train dataset
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsize)
db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
db_val = db_val.shuffle(1000).map(preprocess).batch(batchsize)
```

使用函数`preprocess`进行数据预处理，并数据集进行切片和构建的过程。

`preprocess`函数解释如下：

- 输入参数：`x`表示图像路径，`y`表示标签。
- `x = tf.io.read_file(x)`：使用TensorFlow的`tf.io.read_file`函数读取图像文件的二进制内容。
- `x = tf.image.decode_jpeg(x, channels=3)`：将图像的二进制内容解码为JPEG图像，并指定通道数为3，得到RGB图像。
- `x = tf.image.resize(x, [112, 96])`：将图像调整为指定的尺寸[112, 96]。
- `x = tf.image.random_flip_left_right(x)`：随机左右翻转图像，增加数据的多样性。
- `x = (tf.cast(x, dtype=tf.float32) - 127.5) / 128.0`：将图像的像素值从[0, 255]范围缩放到[-1, 1]范围，以便于模型的训练。
- `y = tf.convert_to_tensor(y)`：将标签转换为张量格式。
- `y = tf.one_hot(y, depth=class_num)`：将标签进行独热编码，使其具有与类别数量`class_num`相同的维度。

`db_train`和`db_val`的构建过程解释如下：

- `train_image, val_image, train_label, val_lable = load_dataset()`：调用之前定义的`load_dataset`函数获取训练集和验证集的图像路径和标签。
- `class_num = len(np.unique(train_label))`：计算类别数量，即训练集中不重复的标签数量。
- `db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))`：将训练集的图像路径和标签转换为TensorFlow的`Dataset`对象。
- `db_train = db_train.shuffle(1000).map(preprocess).batch(batchsize)`：对训练集进行数据切片，首先进行随机打乱（shuffle），然后对每个样本应用`preprocess`函数进行预处理，最后按批次（batch）进行组织。
- `db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))`：将验证集的图像路径和标签转换为TensorFlow的`Dataset`对象。
- `db_val = db_val.shuffle(1000).map(preprocess).batch(batchsize)`：对验证集进行数据切片，包括随机打乱、预处理和按批次组织。

最终，`db_train`和`db_val`分别表示经过预处理并按批次划分的训练集和验证集的数据集对象，可以用于模型的训练和验证。

### 2)调用网络模型，可预加载模型参数、设置优化器、损失等函数、设置回调函数，以及模型拟合和保存


```python
def mobilefacenet_train():
    model = Mobilefacenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=class_num)
    model.load_weights('pre_weight/mobilefacenet_model.h5', skip_mismatch=True, by_name=True)
    # # 优化器
    # optimizer = Adam(lr=0.001, epsilon=1e-8)
    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    # 模型损失
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 回调函数
    callback_list = [ModelCheckpoint("checkpoints/mobilenet_v2/ep{epoch:02d}-accuracy{accuracy:.3f}-loss{loss:.3f}.h5",
                                     monitor='val_loss',save_weights_only=True,
                                     verbose=1, save_best_only=False, period=2),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
                     TensorBoard(log_dir='logs/mobilenet_v2')]

    # 模型训练
    model.fit(db_train,
              validation_data=db_val,
              validation_freq=1,
              epochs=40, callbacks=callback_list,
              initial_epoch=0)

    # 待完成，lwf数据集上验证
    # 模型保存[输出倒数第三层数据，人脸特征]
    inference_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    inference_model.save('model_data/mobilenet_v2/mobilefacenet_model.h5')
    return model


if __name__ == '__main__':
    mobilefacenet_train()
```




- `model = Mobilefacenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=class_num)`：根据给定的输入形状、特征维度和类别数量，创建Mobilefacenet模型。该模型结构在之前的代码中已定义。
- `model.load_weights('pre_weight/mobilefacenet_model.h5', skip_mismatch=True, by_name=True)`：从预训练的权重文件加载模型的参数。这里使用`load_weights`方法加载预训练的权重文件。
- `optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)`：选择优化器为随机梯度下降（SGD），设置学习率、动量和Nesterov动量的参数。
- `model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])`：编译模型，设置优化器、损失函数和评估指标。这里使用分类交叉熵作为损失函数，使用准确率作为评估指标。
- `callback_list`：定义了一系列回调函数，包括`ModelCheckpoint`用于保存模型的检查点，`ReduceLROnPlateau`用于在验证集上监测指标并调整学习率，以及`TensorBoard`用于可视化训练过程。
- `model.fit`：使用给定的训练集和验证集进行模型训练。指定了训练集和验证集的数据集对象`db_train`和`db_val`，设置了训练的周期数、回调函数等参数。
- `inference_model = Model(inputs=model.input, outputs=model.layers[-3].output)`：创建一个新的推理模型，将输入设置为原始模型的输入，输出设置为原始模型倒数第三层的输出。这里的目的是将原始模型的中间层作为人脸特征提取器进行保存。
- `inference_model.save('model_data/mobilenet_v2/mobilefacenet_model.h5')`：保存推理模型，将其用于之后的人脸特征提取。
- `return model`：返回训练好的模型。

在`if __name__ == '__main__':`的部分，调用`mobilefacenet_train`函数来开始模型的训练过程。

## 3 – 模型测试


```python
 # test result 计算余弦距离
    embedding_yzy = embedding_yzy / np.expand_dims(np.sqrt(np.sum(np.power(embedding_yzy, 2), 1)), 1)
    embedding_lm = embedding_lm / np.expand_dims(np.sqrt(np.sum(np.power(embedding_lm, 2), 1)), 1)
    embedding_zt = embedding_zt / np.expand_dims(np.sqrt(np.sum(np.power(embedding_zt, 2), 1)), 1)
    embedding_ly = embedding_zt / np.expand_dims(np.sqrt(np.sum(np.power(embedding_ly, 2), 1)), 1)
    embedding_test = embedding_test / np.expand_dims(np.sqrt(np.sum(np.power(embedding_test, 2), 1)), 1)

    # get result
    print(np.sum(np.multiply(embedding_yzy, embedding_test), 1))
    print(np.sum(np.multiply(embedding_lm, embedding_test), 1))
    print(np.sum(np.multiply(embedding_zt, embedding_test), 1))
    print(np.sum(np.multiply(embedding_ly, embedding_test), 1))
    print("over")
    # # save database
    # db = np.concatenate((embedding_yzy, embedding_lm, embedding_steve), axis=0)
    # print(db.shape)
    # np.save("pretrained_model/db", db)
```







计算了测试结果的余弦距离。

首先，对每个嵌入向量进行了归一化处理，使其长度为1。这是通过除以嵌入向量的模长来实现的。具体而言，对于每个嵌入向量`embedding_yzy`，计算了其模长，并将每个维度的值除以模长，从而实现了归一化。

接下来，通过计算测试向量与各个嵌入向量之间的点积，得到了余弦相似度。点积可以通过使用`np.sum(np.multiply(embedding1, embedding2), 1)`来计算，其中`embedding1`是一个嵌入向量，`embedding2`是另一个嵌入向量。

最后，打印了每个测试向量与各个嵌入向量的余弦相似度。通过比较余弦相似度的值，可以得出测试向量与嵌入向量的相似程度。

还可以进行保存数据库的操作，将嵌入向量拼接起来并保存为一个数据库文件。

