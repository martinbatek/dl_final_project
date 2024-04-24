# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os
from IPython.display import Audio, display

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Reshape,BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Mean
from tensorflow.keras.datasets import mnist
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
tfpd = tfp.distributions
tfpl = tfp.layers
tfpb = tfp.bijectors


'''
# Make  model checkpoint directory
path_checkpoints_4 = "model_checkpoints/q3a/model_4"
if not os.path.exists(path_checkpoints_4):
    os.makedirs(path_checkpoints_4)
    
# Make model backup directory
path_backup_4 = "model_backup/q3a/model_4"
if not os.path.exists(path_backup_4):
    os.makedirs(path_backup_4)

# Delete log file if it exists
if os.path.exists('training_logs/model_4.csv'):
    os.remove('training_logs/model_4.csv')

# clear model checkpoints directory
for file in os.listdir(path_checkpoints_4):
    if os.path.isfile(file):
        of.remove(path_checkpoints_4+'/'+file)

# Define callback for Backing up and restoring training state in case of interruption
backup_restore_4 = tf.keras.callbacks.BackupAndRestore(path_backup_4)

# Define callback for logging training results
csv_logging_4 = tf.keras.callbacks.CSVLogger('training_logs/model_4.csv',append=True) #Setting append to true in case of interruptions

# Define callback for saving the best model
model_checkpoints_4 = tf.keras.callbacks.ModelCheckpoint(
    filepath=path_checkpoints_4+"/epoch_{epoch}.weights.h5",
    save_best_only=False,
    save_weights_only=True
)

# Define adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
'''
### EXACT SAME ARCHITECHTURE AS MODEL 2
# Construct RVQVector Quantizer layer with embeddings initialized using the Glorot Normal distribution
class RVQVectorQuantizer2(Layer):
    def __init__(self, num_embeddings, embedding_dim, rvq_layers=1, beta=0.25, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.rvq_layers = rvq_layers

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # The gamma parameter will be set to 0.99 as per the instructions
        self.gamma = gamma
        
        # Initialize the embeddings which we will quantize.
        w_init = tf.keras.initializers.GlorotNormal()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings, self.rvq_layers), dtype="float32"
            ),
            trainable=False, # Per the insinuation in the question, now updates independantly of the gradient optimization
            name="embeddings_rvqvae",
        )

        # Initialize cluster counts
        N_init = tf.zeros_initializer()
        self.N = tf.Variable(
            initial_value = N_init(
                shape = (self.num_embeddings,self.rvq_layers), dtype="float32"
            ),
            trainable=False,
            name="cluster_count_rvqvae"
        )

        # Initialize the encoding sum
        self.m = tf.Variable(
            initial_value = self.embeddings,
            trainable=False,
            name="encoding_sum_rvqvae"
        )
        
    def call(self, x, training=None):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Residual Vector Quantization
        quantized = tf.zeros(tf.shape(flattened))
        residual = flattened
        encoding_list = []
        update_list = []
        for layer in range(self.rvq_layers):
            encoding_indices = self.get_code_indices(residual,layer)
            encodings = tf.one_hot(encoding_indices, self.num_embeddings)
            encoding_list.append(encodings)
            update = tf.matmul(encodings, self.embeddings[...,layer], transpose_b=True)
            update_list.append(update)
            quantized+=update
            residual-=update

        # Exponential moving average updates to the embeddings during training
        if training:
            encodings = tf.stack(encoding_list,axis=-1)
            quantizations = tf.stack(update_list,axis=-1)
            n = tf.math.reduce_sum(encodings,axis=0)

            m_list = []
            for layer in range(self.rvq_layers):
                E_sum = tf.matmul(quantizations[...,layer],encodings[...,layer],transpose_a=True)
                m_list.append(self.gamma*self.m[...,layer] + (1 - self.gamma)*E_sum)
            
            N_update = self.gamma*self.N + (1 - self.gamma)*n
            m_update = tf.stack(m_list,axis=-1)

            self.N.assign(N_update)
            self.m.assign(m_update)
            embedding_update = tf.math.multiply(self.m,tf.math.reciprocal_no_nan(self.N))
            self.embeddings.assign(embedding_update)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        self.add_loss(self.beta * commitment_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs,layer):
        # Calculate L2-normalized distance between the inputs and the codes.
        embeddings = self.embeddings[...,layer]
        similarity = tf.matmul(flattened_inputs, embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

# Define functions for the encoder and decoder
def get_encoder(latent_dim=16,input_shape=(28,28,1)):
    encoder_inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = Conv2D(latent_dim, 1, padding="same")(x)
    return Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16,input_shape = (28,28,1)):
    latent_inputs = Input(shape=get_encoder(latent_dim,input_shape).output.shape[1:])
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(input_shape[2], 3, padding="same")(x)
    return Model(latent_inputs, decoder_outputs, name="decoder")

# Define function to compose the RVQ-VAE model
def get_rvqvae2(input_shape=(28,28,1),latent_dim=16, num_embeddings=64,rvq_layers=3):
    rvq_layer = RVQVectorQuantizer2(num_embeddings, latent_dim, rvq_layers, name="residual_vector_quantizer")
    encoder = get_encoder(latent_dim,input_shape)
    decoder = get_decoder(latent_dim,input_shape)
    inputs = Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = rvq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return Model(inputs, reconstructions, name="rvq_vae")

# Wrap training loop in RVQVAETrainer
class RVQVAETrainer2(Model):
    def __init__(self, train_variance,shape=(28,28,1), latent_dim=32, num_embeddings=128, rvq_layers=3,gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.encoder = get_encoder(latent_dim,shape)
        self.rvq_layer = RVQVectorQuantizer2(num_embeddings, latent_dim, rvq_layers, name="residual_vector_quantizer",gamma=gamma)
        self.decoder = get_decoder(latent_dim,shape)

        self.total_loss_tracker = Mean(name="loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.rvq_loss_tracker = Mean(name="rvqvae_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.rvq_loss_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the RVQ-VAE.
            encoder_output = self.encoder(x,training=True)
            rvq_layer_output = self.rvq_layer(encoder_output,training=True)
            reconstructions = self.decoder(rvq_layer_output,training=True)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.rvq_layer.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.rvq_loss_tracker.update_state(sum(self.rvq_layer.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "rvqvae_loss": self.rvq_loss_tracker.result(),
        }

    def test_step(self,x):
        # Get the outputs from the RVQ-VAE model
        encoder_output = self.encoder(x)
        rvq_layer_output = self.rvq_layer(encoder_output)
        reconstructions = self.decoder(rvq_layer_output)

        # Calculate the losses
        reconstruction_loss = (
            tf.reduce_mean((x - reconstructions)**2/self.train_variance) # Monte Carlo estimate of E[p(x|z)], assuming Normal p(x|z)
        )
        rvqvae_loss = sum(self.rvq_layer.losses)
        total_loss = reconstruction_loss + rvqvae_loss

        # Loss tracking
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.rvq_loss_tracker.update_state(rvqvae_loss)

        # Log results
        return {
            "loss":self.total_loss_tracker.result(),
            "reconstruction_loss":self.reconstruction_loss_tracker.result(),
            "rvqvae_loss":self.rvq_loss_tracker.result()
        }
'''
### COMPILE AND FIT WITH INCREASED GAMMA
# Compile and fit the model
rvqvae_trainer4 = RVQVAETrainer2(train_var,shape=(124,124,2),latent_dim=16,num_embeddings=128,rvq_layers=5,gamma=0.999)
rvqvae_trainer4.compile(optimizer=optimizer,loss='loss')
model_4_hist = rvqvae_trainer4.fit(
    train_ds,
    validation_data = val_ds,
    epochs=40,
    callbacks=[backup_restore_4,csv_logging_4,model_checkpoints_4]
)
'''