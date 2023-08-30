
from tensorflow import Tensor
from tensorflow.keras.models import Model
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

MODEL_NAME = 'mob_v1_slice'
EXIT_AFTER_CREATING_THE_FIRST = True

dw_repititions = 0
pw_repititions = 100

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
          strides=(2, 2), input_shape=(224, 224, 3), padding='same', activation='relu'))

for i in range(dw_repititions):
    model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'))

for i in range(pw_repititions):
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))


# model.add(layers.DepthwiseConv2D((3, 3), strides=(
#     2, 2), padding='same', activation='relu'))
# model.add(layers.Conv2D(128, (1, 1), activation='relu'))
# model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'))
# model.add(layers.Conv2D(128, (1, 1), activation='relu'))
# model.add(layers.DepthwiseConv2D((3, 3), strides=(
#     2, 2), padding='same', activation='relu'))

print(model.summary())

model.save(MODEL_NAME + "_inout")

# to use later by trtexec:
# first use python3 -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
# e.g. python3 -m tf2onnx.convert --saved-model uniform_mobilenetv2_75_32_inout --output uniform_mobilenetv2_75.onnx
# this will convert the model to onnx that can be used by trtexec but not trt scripts
# second: run trtexec and dump the output as trt engine:
# ./trtexec --onnx=onnx_model_path --int8 --saveEngine=path_to_save_trt_engine
# third: run the resulte using trt scripts

if EXIT_AFTER_CREATING_THE_FIRST:
    exit(0)
############################################################################
MODEL_NAME = 'mob_v1_slice_h'

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
          strides=(2, 2), input_shape=(224, 16, 3), padding='same', activation='relu'))
model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (1, 1), activation='relu'))
model.add(layers.DepthwiseConv2D((3, 3), strides=(
    2, 2), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (1, 1), activation='relu'))
model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (1, 1), activation='relu'))
model.add(layers.DepthwiseConv2D((3, 3), strides=(
    2, 2), padding='same', activation='relu'))

print(model.summary())

model.save(MODEL_NAME + "_inout")

############################################################################


MODEL_NAME = 'resnet50_slice'


def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2D(kernel_size=1,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(x)
    #y = relu_bn(y)

    y = layers.Conv2D(kernel_size=kernel_size,
                      strides=1,
                      filters=filters,
                      padding="same")(y)

    y = layers.Conv2D(kernel_size=1,
                      strides=1,
                      filters=filters * 4,
                      padding="same")(y)
    #y = relu_bn(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1,
                          strides=2,
                          filters=filters * 4,
                          padding="same")(x)
    out = layers.Add()([x, y])
    #out = relu_bn(out)
    return out


def create_res_net(input_shape=(224, 224, 3), num_blocks_list = [2, 5, 5, 2]):

    inputs = layers.Input(shape=input_shape)
    num_filters = 64

    t = layers.BatchNormalization()(inputs)
    t = layers.Conv2D(kernel_size=7,
                      strides=2,
                      filters=num_filters,
                      padding="same")(t)
    #t = relu_bn(t)

    #num_blocks_list = [2, 5]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(
                j == 0), filters=num_filters)
        num_filters *= 2

    # t = layers.AveragePooling2D(4)(t)
    # t = layers.Flatten()(t)
    outputs = layers.Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)

    return model


model = create_res_net(num_blocks_list = [2, 5])

print(model.summary())

model.save(MODEL_NAME + "_inout")

############################################################################
MODEL_NAME = 'resnet50_slice_h'

model = create_res_net(input_shape=(224, 16, 3), num_blocks_list = [2, 5])

print(model.summary())

model.save(MODEL_NAME + "_inout")

############################################################################
MODEL_NAME = 'resnet50_full'

model = create_res_net(input_shape=(224, 224, 3))

print(model.summary())

model.save(MODEL_NAME + "_inout")