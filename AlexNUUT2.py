import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Paths to your dataset
train_dir = r"C:\Users\Javier\Downloads\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train"
test_dir = r"C:\Users\Javier\Downloads\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\test"

# Define AlexNet model
def create_alexnet(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Assuming 227x227 input size and 1000 classes (adjust as necessary)
input_shape = (227, 227, 3)
num_classes = 1000
model = create_alexnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create ImageDataGenerators for loading images from directories
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images in batches from the train and test directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=16,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode='sparse'
)

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# Confusion Matrix (modify as needed for batch-based data)
pred_labels = []
true_labels = []
for images, labels in test_generator:
    predictions = model.predict(images)
    pred_labels.extend(np.argmax(predictions, axis=1))
    true_labels.extend(labels)
    if len(true_labels) >= test_generator.samples:  # Stop after one pass over test set
        break

conf_matrix = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.colorbar()
plt.show()

# ROC Curve (for binary classification)
if num_classes == 2:
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Show and print a prediction from the test set
sample_images, sample_labels = next(test_generator)
sample_image = sample_images[0]
sample_label = sample_labels[0]
pred_probabilities = model.predict(sample_image[np.newaxis, ...])
predicted_class = np.argmax(pred_probabilities)

plt.imshow(sample_image)
plt.title(f"Predicted: {predicted_class}, Actual: {sample_label}")
plt.show()

print(f"Predicted probabilities: {pred_probabilities}")
print(f"Predicted class: {predicted_class}")
print(f"True class: {sample_label}")
