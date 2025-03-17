import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Custom Network Class (as per your original code)
class CustomNetwork(tf.keras.Model):

    def __init__(self, model_specs):
        super(CustomNetwork, self).__init__()
        layer_sizes = model_specs[0]
        self.layer_list = []
        for i in range(len(layer_sizes) - 1):
            dense_layer = tf.keras.layers.Dense(layer_sizes[i+1],
                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),  
                                                 bias_initializer=tf.keras.initializers.GlorotUniform())
            self.layer_list.append(dense_layer)
        self.build((None, layer_sizes[0]))
    
    def build(self, input_shape):
        self.layer_list[0].build(input_shape)
        for i in range(1, len(self.layer_list)):
            self.layer_list[i].build((input_shape[0], self.layer_list[i-1].units))
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list[:-1]:
            x = tf.keras.activations.relu(layer(x))
        x = tf.keras.activations.sigmoid(self.layer_list[-1](x))
        return x
    
    def compute_loss(self, Y, Y_hat):
        L_sum = 0.5 * tf.reduce_sum(tf.square(Y - Y_hat))
        m = tf.cast(tf.shape(Y)[0], tf.float32)
        L = (1. / m) * L_sum
        return L

    def get_obj(self, x, y, params):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)   
        model_parameters = self.trainable_variables
        for i, param in enumerate(params):
            model_parameters[i].assign(param)
        obj_fwd = tf.reshape(self.call(x), [-1])
        fval = self.compute_loss(obj_fwd, tf.reshape(y, [-1]))
        return fval.numpy()

    def get_obj_grad(self, x, y, params):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)   
        with tf.GradientTape() as tape:
            obj_fwd = tf.reshape(self.call(x),[-1])
            obj_loss = self.compute_loss(obj_fwd, tf.reshape(y, [-1]))
        gradients = tape.gradient(obj_loss, self.trainable_variables)
        fgrad = tf.concat([tf.reshape(grad, [-1]) for grad in gradients if grad is not None], axis=0)
        return fgrad

    def save_model(self, dir):
        self.save(str(dir)+'.h5')

    def evaluate(self, x):
        return self.call(x).numpy()

# Streamlit UI setup
def run():
    st.title('Custom Neural Network Model')

    # Input section
    st.subheader("Enter Model Specifications")
    input_dim = st.number_input("Input dimension (number of features)", min_value=1, value=10, step=1)
    hidden_units = st.number_input("Hidden Layer Size", min_value=1, value=64, step=1)
    output_dim = st.number_input("Output dimension", min_value=1, value=1, step=1)

    if st.button('Generate Model'):
        model_specs = [[input_dim, hidden_units, output_dim]]
        model = CustomNetwork(model_specs)
        st.write("Model Generated!")

    # Input data section
    st.subheader("Input Data")
    num_samples = st.number_input("Number of samples", min_value=1, value=100, step=1)

    # Generate random data
    if st.button("Generate Data"):
        X_train = np.random.rand(num_samples, input_dim)
        y_train = np.random.randint(0, 2, size=(num_samples, output_dim))
        st.write("Data generated!")
        st.write("X_train shape:", X_train.shape)
        st.write("y_train shape:", y_train.shape)

    # Model training section
    if st.button("Train Model"):
        X_train = np.random.rand(num_samples, input_dim)  # Simulated data for training
        y_train = np.random.randint(0, 2, size=(num_samples, output_dim))  # Simulated binary targets
        
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        st.write("Training complete!")

    # Evaluate section
    if st.button("Evaluate Model"):
        X_train = np.random.rand(num_samples, input_dim)  # Simulated data for evaluation
        predictions = model.evaluate(X_train)
        st.write("Model evaluation complete!")
        st.write("Predictions: ", predictions)

    # Save Model
    if st.button("Save Model"):
        save_path = st.text_input("Enter model save directory:", "./saved_model")
        model.save_model(save_path)
        st.write(f"Model saved to {save_path}.h5")

# Run the UI
if __name__ == "__main__":
    run()
