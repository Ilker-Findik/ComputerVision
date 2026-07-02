from pathlib import Path

from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[1] / "Datasets"

(train_dataset, test_dataset), info = load(
    "tf_flowers", 
    split=["train[:80%]", "train[80%:]"], 
    as_supervised=True,
    with_info=True,
    data_dir=str(DATA_DIR),
    download=True  
)



print(info)
print("Number of classes:", info.features["label"].num_classes)

# fig = plt.figure(figsize=(10, 5))
# for i, (img, label) in enumerate(train_dataset.take(3)):
#     ax = fig.add_subplot(1, 5, i + 1)
#     ax.imshow(img.numpy().astype("uint8"))
#     ax.set_title(info.features["label"].int2str(label))
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

IMG_SIZE = (180,180)

def preprocess_train(image, label):
    image =tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2)
    image = tf.image.random_crop(image, size=[160,160,3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_test(image, label):
    img = tf.image.resize(image, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_dataset = (train_dataset
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(AUTOTUNE)
)

test_dataset = (test_dataset
    .map(preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),



    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(info.features["label"].num_classes, activation='softmax')
])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,verbose = 1,min_lr=1e-9),
    ModelCheckpoint(str(Path(__file__).resolve().with_name('best_model.keras')), save_best_only=True)
]

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_dataset, 
    validation_data=test_dataset, 
    epochs=2, 
    callbacks=callbacks
)