import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from vit_keras import vit

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def my_vit_l32(image_size, **kwargs):
    if isinstance(image_size, tuple):
        image_size = max(image_size)
    return vit.vit_l32(image_size=image_size, **kwargs)

model_vit = Sequential([
    Input(shape=(32, 32, 3)),
    my_vit_l32(
        image_size=(32, 32),
        classes=10,
        activation='softmax',
        pretrained=True
    )
])

model_vit.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model_vit.summary()

model_vit.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss_vit, test_acc_vit = model_vit.evaluate(x_test, y_test)
print(f'ViT Test accuracy: {test_acc_vit * 100:.2f}%')
