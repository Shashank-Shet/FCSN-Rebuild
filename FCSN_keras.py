#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPool1D,
    Dropout,
    ReLU,
    Conv2DTranspose,
    Input,
    Add,
    Layer,
    Softmax
)
from tensorflow.keras import Model


# In[2]:


class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        self.output_channels = filters
        self.kernel_size = (kernel_size, 1)
        self.strides = (stride, 1)
        self.kwargs = kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        self.layer = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            **self.kwargs
        )
        self.layer.build((input_shape[0], input_shape[1], 1, input_shape[2]))
        self._trainable_weights = self.layer.trainable_weights
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        data = K.expand_dims(x, axis=2)
        data = self.layer(data)
        data = K.squeeze(data, axis=2)
        return data


# # FCSN Model

# In[11]:


input_size = (320, 1024)   # Tensorflow uses the Channels-last format by default
n_classes  = 2

inputs = Input(input_size)

# Block 1
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool1D(pool_size=2, strides=2, padding="SAME")(x)


# In[12]:


# Block 2
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool1D(pool_size=2, strides=2, padding="SAME")(x)


# In[13]:


# Block 3
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=1024, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool1D(pool_size=2, strides=2, padding="SAME")(x)


# In[14]:


# Block 4
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool1D(pool_size=2, strides=2, padding="SAME")(x)

pool4 = x


# In[15]:


# Block 5
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=2048, kernel_size=3, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool1D(pool_size=2, strides=2, padding="SAME")(x)


# In[16]:


# Block 6
x = Conv1D(filters=4096, kernel_size=1, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.5)(x)

# Block 7
x = Conv1D(filters=4096, kernel_size=1, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.5)(x)

# Block 8
x = Conv1D(filters=n_classes, kernel_size=1, padding="SAME")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv1DTranspose(filters=n_classes, kernel_size=4, padding="SAME", stride=2, use_bias=False)(x)

upscore = x


# In[17]:


score_pool = Conv1D(filters=n_classes, kernel_size=1, padding="SAME")(pool4)
score_pool = BatchNormalization()(score_pool)

x = Add()([upscore, score_pool])
x = Conv1DTranspose(filters=n_classes, kernel_size=16, padding="SAME", stride=16, use_bias=False)(x)

outputs = Softmax()(x)

model = Model(inputs=inputs, outputs=outputs, name="FCSN")


# In[18]:


model.summary()


# In[21]:


model.compile(
    loss=tf.keras.losses.categorical_crossentropy, 
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

