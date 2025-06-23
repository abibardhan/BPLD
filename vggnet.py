import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your training and testing directories
train_dir = 'train_folder'
test_dir = 'test_folder'

# Define image size and batch size
img_size = (150, 150)  # Ensure this matches the input size for VGG
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical for multi-class classification
)

# Load the testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical for multi-class classification
)

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

vgg_base.trainable = False

# Build the model
model = models.Sequential([
    vgg_base,  # Use the pre-trained VGG16 base
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Adjust to match the number of classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 25  # Adjust based on your dataset size and complexity

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Save the model
model.save('vgg16_finetuned_model.keras')