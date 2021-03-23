#import libraries 
import os 
import pathlib

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# load data
data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

#check data
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands!='README.md']
print("our commands are: ", commands)

# split audio files and shuffle them
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
number_of_samples = len(filenames)
print("the number of audios", number_of_samples)
print("number examples per labels", len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print("our first exampe is: ", filenames[0])

# split data for training and testing
train_files = filenames[:6400]
validation_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]
print("train example size is: ", len(train_files))
print("validation example size is: ", len(validation_files))
print("test example size is: ", len(test_files))

#reading audio files nd their labels
def decode_audio(audio_binary):
    audio,_ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
wavefrom_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# to get spectogram
def get_spectogram(waveform):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype = tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step = 128)
    spectogram = tf.abs(spectogram)
    return spectogram

for waveform, label in wavefrom_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectogram = get_spectogram(waveform)

print("our labelis: ", label)
print("waveform shape is: ", waveform.shape)
print("spectogram shape is: ", spectogram.shape)
print("audio playback")
display.display(display.Audio(waveform, rate=16000))

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = wavefrom_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectogram_and_label_id, num_parallel_calls = AUTOTUNE)
    return output_ds

train_ds = spectogram_ds
val_ds = preprocess_dataset(validation_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32), 
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')
