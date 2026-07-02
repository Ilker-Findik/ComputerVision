import numpy as np
import cv2 
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img = X_train[12]

stages = {'original': img}

eq = cv2.equalizeHist(img)
stages['equalized'] = eq

blur = cv2.GaussianBlur(eq, (5, 5), 0)
stages['blurred'] = blur

edges = cv2.Canny(blur, 50, 150)
stages['edges'] = edges

# fig,axes = plt.subplots(1, 4, figsize=(6, 6))
# axes = axes.flat
# for ax, (title,img) in zip(axes, stages.items()):
#     ax.imshow(img, cmap='gray')
#     ax.set_title(title)
#     ax.axis('off')

# plt.suptitle('Image Processing Stages')
# plt.tight_layout()
# plt.show()


def preprocess_images(img):
    eq = cv2.equalizeHist(img)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    features=edges.flatten()/255.0
    return features

num_train = 10000
num_test = 2000

X_train = np.array([preprocess_images(img) for img in X_train[:num_train]])
y_train = y_train[:num_train]

x_test = np.array([preprocess_images(img) for img in X_test[:num_test]])
y_test = y_test[:num_test]

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    batch_size=32,
    validation_split=0.2
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')  
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  
plt.show()