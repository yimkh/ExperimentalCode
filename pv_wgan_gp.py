import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, LSTM, Dropout, Bidirectional, GaussianNoise, \
    MultiHeadAttention, Input
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

PATH = './Save/'
data_NAME = 'data'
name = 'Temperature'
PATH = PATH + data_NAME + name + '/'
os.makedirs(PATH, exist_ok=True)

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    pv_data = df[name].values

    pv_data = np.maximum(pv_data, 0)
    pv_data = np.log1p(pv_data)

    scaler = MinMaxScaler()
    pv_scaled = scaler.fit_transform(pv_data.reshape(-1, 1))

    def create_sequences(data, seq_length, output_length):
        X, y = [], []
        for i in range(len(data) - seq_length - output_length + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + output_length])
        return np.array(X), np.array(y)

    seq_length = 20
    output_length = 4
    X, y = create_sequences(pv_scaled, seq_length, output_length)

    return X, y, scaler, pv_data[seq_length + output_length - 1:], df['timestamp'].values[
                                                                   seq_length + output_length - 1:]


def make_generator_model(input_dim, output_dim, feature_size):
    inputs = Input(shape=(input_dim, feature_size))

    # BiLSTM
    x = Bidirectional(LSTM(units=512, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(units=256, return_sequences=True))(x)

    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(query=x, value=x, key=x)

    x = Flatten()(attention_output)
    x = Dense(128)(x)
    x = GaussianNoise(0.05)(x)
    x = Dense(64)(x)
    x = Dense(units=output_dim * 1)(x)
    x = tf.keras.layers.Reshape((output_dim, 1))(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def make_discriminator_model():
    model = Sequential([
        Conv1D(32, kernel_size=3, strides=2, padding='same', input_shape=(4, 1)),
        LeakyReLU(alpha=0.01),
        Conv1D(64, kernel_size=5, strides=2, padding='same'),
        LeakyReLU(alpha=0.01),
        Conv1D(128, kernel_size=5, strides=2, padding='same'),
        LeakyReLU(alpha=0.01),
        Flatten(),
        Dense(220, use_bias=False),
        LeakyReLU(),
        Dense(220, use_bias=False),
        Dense(1)
    ])
    return model


# WGGP
class WGGP:
    def __init__(self, generator, discriminator, opt):
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator = discriminator
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999)
        self.batch_size = opt['bs']
        self.lambda_gp = 5.0
        self.n_critic = 3
        self.checkpoint_dir = './training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = real_data + alpha * (fake_data - real_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def discriminator_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_loss(self, fake_output, real_y, generated_data, lambda_mse=2.0):
        gan_loss = -tf.reduce_mean(fake_output)
        mse_loss = self.mse_loss(real_y, generated_data)
        return gan_loss + lambda_mse * mse_loss

    def train_step(self, real_x, real_y):
        for _ in range(self.n_critic):
            with tf.GradientTape() as disc_tape:
                generated_data = self.generator(real_x, training=True)
                real_y_reshape = tf.cast(real_y, dtype=tf.float32)
                real_output = self.discriminator(real_y_reshape, training=True)
                fake_output = self.discriminator(generated_data, training=True)
                disc_loss = self.discriminator_loss(real_output, fake_output)
                gp = self.gradient_penalty(real_y_reshape, generated_data)
                total_disc_loss = disc_loss + self.lambda_gp * gp

            gradients_of_discriminator = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(real_x, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output, real_y, generated_data)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, total_disc_loss

    def train(self, real_x, real_y, epochs):
        gen_losses, disc_losses = [], []
        for epoch in range(epochs):
            start = time.time()
            gen_loss, disc_loss = self.train_step(real_x, real_y)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(
                    f'Epoch {epoch + 1}, gen_loss: {gen_loss}, disc_loss: {disc_loss}, time: {time.time() - start:.2f}s')
        plt.figure(figsize=(10, 5))
        plt.plot(gen_losses, label="Generator Loss")
        plt.plot(disc_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(PATH, "loss_plot.png"))
        plt.show()
        return self.generator

def plot_and_evaluate(real_data, generated_data, scaler, timestamps, set_name="Generated"):
    real_last = real_data[:, -1]
    generated_last = generated_data[:, -1]

    rescaled_real = np.expm1(scaler.inverse_transform(real_last.reshape(-1, 1)))
    rescaled_generated = np.expm1(scaler.inverse_transform(tf.reshape(generated_last, (-1, 1)).numpy()))

    print("Rescaled real stats:", rescaled_real.min(), rescaled_real.max(), rescaled_real.mean())
    print("Rescaled generated stats:", rescaled_generated.min(), rescaled_generated.max(), rescaled_generated.mean())

    result_df = pd.DataFrame({
        "Timestamp": timestamps[:len(rescaled_real)],
        "Real": rescaled_real.flatten(),
        "Generated": rescaled_generated.flatten()
    })
    result_file_path = PATH + data_NAME + f"_{set_name}_results.xlsx"
    result_df.to_excel(result_file_path, index=False)
    print(f"result {result_file_path}")

    rmse = np.sqrt(mean_squared_error(rescaled_real, rescaled_generated))
    print(f"{set_name} RMSE: {rmse}")

    plt.figure(figsize=(16, 8))
    dates = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps[:len(rescaled_real)]]
    plt.plot(dates, rescaled_real, label="Real")
    plt.plot(dates, rescaled_generated, label="Generated", color='red')
    plt.xlabel("Timestamp")
    plt.ylabel(name)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.grid(True)
    image_file_path = PATH + data_NAME + f"_{set_name}_plot.png"
    plt.savefig(image_file_path)
    print(f"picture {image_file_path}")
    plt.show()

    return rmse

if __name__ == '__main__':
    file_path = data_NAME + '.xlsx'
    X_full, y_full, scaler, real_full_data, timestamps = load_and_preprocess_data(file_path)

    input_dim, feature_size, output_dim = X_full.shape[1], X_full.shape[2], 4
    opt = {"lr": 0.0001, "epoch": 60, "bs": 64}

    generator = make_generator_model(input_dim, output_dim, feature_size)
    discriminator = make_discriminator_model()
    wggp = WGGP(generator, discriminator, opt)

    trained_generator = wggp.train(X_full, y_full, opt["epoch"])
    generated_full = trained_generator(X_full)
    plot_and_evaluate(y_full, generated_full, scaler, timestamps, set_name="Full")
