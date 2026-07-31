import tensorflow as tf

# ----------------------------
# TensorFlow Functional API Demo
# ----------------------------

inputs = tf.keras.Input(shape=(224, 224, 3), name="input")

# First Block
x = tf.keras.layers.Conv2D(
    32,
    kernel_size=3,
    padding="same",
    name="conv1"
)(inputs)

x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.ReLU(name="relu1")(x)

# Save skip connection
skip = x

# Second Block
x = tf.keras.layers.Conv2D(
    32,
    kernel_size=3,
    padding="same",
    name="conv2"
)(x)

x = tf.keras.layers.BatchNormalization(name="bn2")(x)

# Residual Connection
x = tf.keras.layers.Add(name="add")([x, skip])

x = tf.keras.layers.ReLU(name="relu2")(x)

# Classification Head
x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

outputs = tf.keras.layers.Dense(
    10,
    activation="softmax",
    name="classifier"
)(x)

# Final Model
model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="TensorFlowDemoModel"
)

# Optional: Print summary when executed directly
if __name__ == "__main__":
    model.summary()