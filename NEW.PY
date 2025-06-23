import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121

# Set the paths to your training and testing directories
train_dir = 'train_folder'
test_dir = 'test_folder'

# Define image size and batch size
img_size = (224, 224)  # Adjusted image size
batch_size = 32
num_classes = 5  # Number of output classes (A, H, L, P, Y)

# Create image data generators (without data augmentation)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load DenseNet121 as a base model (pre-trained on ImageNet)
base_model = DenseNet121(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,  # Exclude DenseNet's classification layers
    weights='imagenet'  # Use pre-trained ImageNet weights
)

# Unfreeze the last 10 layers of DenseNet121 for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Squeeze and Excitation (SE) block
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.multiply([input_tensor, se])  # Ensure the input_tensor is returned as a valid output

# Build the DRPA-Net model with SE block
x = base_model.output
x = se_block(x)  # Add SE Block after DenseNet block
x = layers.GlobalAveragePooling2D()(x)  # Global average pooling
x = layers.BatchNormalization()(x)  # Apply Batch Normalization
x = layers.Dense(512, activation='relu')(x)  # Increase the number of units
x = layers.Dropout(0.5)(x)  # Dropout to prevent overfitting
x = layers.Dense(256, activation='relu')(x)  # Additional Dense layer
x = layers.Dropout(0.5)(x)  # Dropout
output = layers.Dense(num_classes, activation='softmax')(x)  # Final output

# Define the model
model = models.Model(inputs=base_model.input, outputs=output)  # Correctly specify input and output

# Compile the model with label smoothing and Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
epochs = 25
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Save the model
model.save('drpa_net_model_finetuned.keras')
