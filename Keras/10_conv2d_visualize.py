######## layer visualization

import keras
from keras.models import load_model


model = load_model("./visual/model.h5")
model.summary()

import numpy as np
from keras.preprocessing import image

img = image.load_img("./visual/1.jpg", target_size=(128, 128))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor = img_tensor / 127.5 - 1

print(img_tensor.shape)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
# plt.show()


from keras import models
# 상위 8개 층의 출력을 추출합니다:
layer_outputs = [layer.output for layer in model.layers[1:8]]
# 입력에 대해 8개 층의 출력을 반환하는 모델을 만듭니다:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 19], cmap="viridis")


plt.matshow(first_layer_activation[0, :, :, 15], cmap="viridis")
plt.show()

####################################  전체 그리기

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            
            
