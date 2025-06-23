import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your training and testing directories
train_dir = 'train_folder'
test_dir = 'test_folder'

# Define image size and batch size
img_size = (150, 150)  # You can change this based on your needs
batch_size = 32

# Create image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
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

# Define a TresNet Block
def tresnet_block(x, filters, strides=1):
    shortcut = x
    
    # Depthwise Separable Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add the shortcut back to the output
    if shortcut.shape[-1] != filters:  # If the number of filters is different, we need to adjust the shortcut
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# Define the TresNet model
input_layer = layers.Input(shape=(img_size[0], img_size[1], 3))

# Initial convolution
x = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2))(input_layer)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# TresNet Blocks
x = tresnet_block(x, 32)
x = tresnet_block(x, 64, strides=2)
x = tresnet_block(x, 128, strides=2)

# Global Average Pooling
x = layers.GlobalAveragePooling2D()(x)
output_layer = layers.Dense(5, activation='softmax')(x)  # 5 classes: A, H, L, P, Y

model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set the number of epochs
epochs = 25  # Adjust based on your dataset size and complexity

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Save the model
model.save('tresnet_model.keras')
