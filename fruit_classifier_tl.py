# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:16:48 2024

@author: tamer
"""

# Importing necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

# Defining paths for dataset
DATASET_PATH = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
TRAIN_PATH = os.path.join(DATASET_PATH, "Training")
VALIDATION_PATH = os.path.join(DATASET_PATH, "Validation")
TEST_PATH = os.path.join(DATASET_PATH, "Test")

# Step 1: Data Preparation
print("Preparing data...")
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Step 2: Model Initialization (Transfer Learning)
print("Initializing pre-trained models...")
pretrained_models = {
    "MobileNetV2": MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "EfficientNetB0": EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "DenseNet121": DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
}

models = {}
for model_name, base_model in pretrained_models.items():
    print(f"Setting up {model_name}...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freezing the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    models[model_name] = model

# Step 3: Training Each Model
EPOCHS = 10
class CustomStopper(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        if acc == 1.0 and val_acc == 1.0:
            print("\nAccuracy and Validation Accuracy have reached 1. Stopping training.")
            self.model.stop_training = True

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Adding custom stopper
    custom_stopper = CustomStopper()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[custom_stopper]
    )

    # Save the model
    model.save(f"{model_name}_fruit_classifier.h5")
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Step 4: Evaluate and Generate Reports
print("Evaluating models...")
results = {}
cam_models = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()))
    results[model_name] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(test_generator.class_indices.keys()), yticklabels=list(test_generator.class_indices.keys()))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Save report
    with open(f"{model_name}_classification_report.txt", "w") as f:
        f.write(report)

    # Save GradCAM models for visualization
    cam_models[model_name] = model

def create_gradcam_visualization(model, img_array, class_idx, layer_name=None):
    """
    Model'in belirli bir sınıf için odaklandığı bölgeleri görselleştirir.
    
    Args:
        model: Eğitilmiş model
        img_array: Görselleştirilecek görüntü (normalized)
        class_idx: Hedef sınıfın indeksi
        layer_name: GradCAM için kullanılacak katman adı (None ise son conv katmanı kullanılır)
    """
    # Layer seçimi
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Score fonksiyonu oluştur
    score = CategoricalScore(class_idx)
    
    # Gradcam nesnesini oluştur
    gradcam = Gradcam(model,
                      model_modifier=None,
                      clone=False)
    
    # Gradcam hesapla
    cam = gradcam(score,
                  img_array,
                  penultimate_layer=layer_name)
    
    # Normalize et
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Görselleştirme
    plt.figure(figsize=(10, 4))
    
    # Orijinal görüntü
    plt.subplot(131)
    plt.imshow(img_array[0])
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(132)
    plt.imshow(heatmap[0], cmap='jet')
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    # Overlap
    plt.subplot(133)
    plt.imshow(img_array[0])
    plt.imshow(heatmap[0], cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Modeliniz için kullanım örneği:
for model_name, model in models.items():
    print(f"\nModel: {model_name} için GradCAM görselleştirmeleri:")
    
    # Test setinden birkaç örnek al
    test_generator.reset()
    for i in range(3):  # İlk 3 görüntü için
        img_batch, label_batch = next(iter(test_generator))
        true_class = np.argmax(label_batch[0])
        
        # Tahmin yap
        pred = model.predict(img_batch)
        pred_class = np.argmax(pred[0])
        
        print(f"\nGörüntü {i+1}")
        print(f"Gerçek sınıf: {list(test_generator.class_indices.keys())[true_class]}")
        print(f"Tahmin edilen sınıf: {list(test_generator.class_indices.keys())[pred_class]}")
        
        # GradCAM görselleştirmesi
        create_gradcam_visualization(model, img_batch, pred_class)
        
print("Process completed.")
