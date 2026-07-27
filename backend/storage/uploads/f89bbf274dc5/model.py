import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Prevent unnecessary log clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print(f"Using TensorFlow version: {tf.__version__}")
    
    # 2. Load and preprocess dummy dataset
    # (Using standard built-in MNIST as an example)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 3. Define the architecture
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    # 4. Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Train the model
    print("Starting training...")
    model.fit(x_train, y_train, epochs=3)

    # 6. Evaluate and save the graph
    model.evaluate(x_test, y_test)
    model.save("my_saved_model.keras")
    print("Model saved successfully as 'my_saved_model.keras'.")

if __name__ == "__main__":
    main()