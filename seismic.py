import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# 1. Data generators
IMG_SIZE = (224, 224)
BATCH = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'data/train', 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    'data/val', 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode='categorical'
)

print(f"Number of classes: {train_gen.num_classes}")
print(f"Class indices: {train_gen.class_indices}")

# 2. Model - Transfer learning with explicit input shape
input_tensor = layers.Input(shape=(224, 224, 3))
base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
base.trainable = False  # freeze for initial training

# Build model using Functional API instead of Sequential
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
predictions = layers.Dense(train_gen.num_classes, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Try model summary - if this fails, we'll skip it and continue training
try:
    model.summary()
except Exception as e:
    print(f"Model summary failed: {e}")
    print("Continuing without summary...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 3. Callbacks
callbacks = [
    ModelCheckpoint('models/seismic_resnet.keras', save_best_only=True, monitor='val_loss'),  # CHANGED to .keras
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# 4. Train
print("Starting training...")
history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

# 5. Fine-tune (unfreeze some layers)
print("Starting fine-tuning...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(train_gen, validation_data=val_gen, epochs=8, callbacks=callbacks)

# 6. Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'data/test', 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode='categorical', 
    shuffle=False
)
res = model.evaluate(test_gen)
print("Test loss/acc:", res)

# 7. Grad-CAM function to interpret predictions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Example usage - use any image from your test folders
try:
    img_path = 'data/test/with_hydrocarbon/with_hydrocarbon_1.png'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(input_arr)
    print("Class probs:", preds)
    heatmap = make_gradcam_heatmap(input_arr, model, last_conv_layer_name='conv5_block3_out')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr)
    plt.imshow(cv2.resize(heatmap, IMG_SIZE), cmap='jet', alpha=0.4)
    plt.title('Grad-CAM overlay')
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Grad-CAM visualization failed: {e}")
