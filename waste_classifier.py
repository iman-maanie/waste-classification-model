# IMPORTS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# PATH SETUP
# Relative path
base_path = os.path.join(os.getcwd(), 'Dataset')
train_dir = os.path.join(base_path, 'Training')
val_dir = os.path.join(base_path, 'Validation')
test_dir = os.path.join(base_path, 'Test sets')

# IMAGE SETTINGS
img_height, img_width = 224, 224
batch_size = 32

# DATA LOADERS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=False
)

# MODEL SETUP
model_path = "my_model.keras"

# Option 1: Load pre-trained model if available
if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)

# Option 2: Train a new model
else:
    print("Training new model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Save the model
    model.save(model_path)
    print(f"Model saved to: {model_path}")

# EVALUATION
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

# PREDICTIONS AND METRICS
pred_probs = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(pred_probs, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Display sample predictions
test_images = []
for i in range(len(test_generator)):
    x_batch, _ = test_generator[i]
    test_images.extend(x_batch)
test_images = np.array(test_images)

num_samples = 10
indices = np.random.choice(len(test_images), num_samples, replace=False)

plt.figure(figsize=(20, 10))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx])
    plt.axis('off')
    true_label = class_labels[true_classes[idx]]
    pred_label = class_labels[predicted_classes[idx]]
    confidence = np.max(pred_probs[idx])
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", color=color)
plt.tight_layout()
plt.show(block=False)
plt.pause(15)
plt.close()

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show(block=False)
plt.pause(15)
plt.close()

# AUC-ROC
true_classes_bin = label_binarize(true_classes, classes=np.arange(len(class_labels)))
auc_score = roc_auc_score(true_classes_bin, pred_probs, average='macro', multi_class='ovr')
print(f"\nAUC-ROC Score (macro average): {auc_score:.4f}")

# LIVE WEBCAM PREDICTION WITH BOUNDING BOXES

print("\nStarting live webcam prediction with bounding boxes. Press 'q' to exit.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_size = (224, 224)
threshold = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    box_color = (255, 0, 0)  # Blue
    thickness = 2

    # --- LEFT bounding box ---
    left_box_start = (50, 100)
    left_box_end = (200, 300)
    cv2.rectangle(frame, left_box_start, left_box_end, box_color, thickness)

    # Crop left region
    left_roi = frame[left_box_start[1]:left_box_end[1], left_box_start[0]:left_box_end[0]]
    left_img = cv2.resize(left_roi, img_size)
    left_img_array = img_to_array(left_img)
    left_img_array = np.expand_dims(left_img_array, axis=0) / 255.0

    # Predict left
    left_preds = model.predict(left_img_array)
    left_index = np.argmax(left_preds)
    left_confidence = np.max(left_preds)
    left_label = class_labels[left_index] if left_confidence >= threshold else "Unknown"
    left_text = f"L: {left_label} ({left_confidence:.2f})"

    # Show left prediction
    cv2.putText(frame, left_text, (left_box_start[0], left_box_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- RIGHT bounding box ---
    right_box_start = (width - 200, 100)
    right_box_end = (width - 50, 300)
    cv2.rectangle(frame, right_box_start, right_box_end, box_color, thickness)

    # Crop right region
    right_roi = frame[right_box_start[1]:right_box_end[1], right_box_start[0]:right_box_end[0]]
    right_img = cv2.resize(right_roi, img_size)
    right_img_array = img_to_array(right_img)
    right_img_array = np.expand_dims(right_img_array, axis=0) / 255.0

    # Predict right
    right_preds = model.predict(right_img_array)
    right_index = np.argmax(right_preds)
    right_confidence = np.max(right_preds)
    right_label = class_labels[right_index] if right_confidence >= threshold else "Unknown"
    right_text = f"R: {right_label} ({right_confidence:.2f})"

    # Show right prediction
    cv2.putText(frame, right_text, (right_box_start[0], right_box_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show final frame
    cv2.imshow("Webcam Prediction (Left & Right Boxes)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()