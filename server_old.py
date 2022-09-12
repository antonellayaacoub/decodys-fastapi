from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

commands = []

model1 = load_model('model\model1.h5')

class_predictions1 = array([
    'banzumoti', 'bluglofrin', 'fepani', 'greinblu' ,'juban', 'kovinba',
 'nidoujuse' ,'plubro', 'sevu' ,'treplabli',  'doujusegani', 'fleublagropriklu' ,'fleuprikrobla', 'greinklofloubra',
 'problifrouklebro','tireuchonkozein'
])
commands1 = ['banzumoti', 'bluglofrin', 'fepani', 'greinblu' ,'juban', 'kovinba',
 'nidoujuse' ,'plubro', 'sevu' ,'treplabli',  'doujusegani', 'fleublagropriklu' ,'fleuprikrobla', 'greinklofloubra',
 'problifrouklebro','tireuchonkozein']


model12 = load_model('model\model12.h5')

class_predictions12 = array([
   'doujusegani', 'fleublagropriklu' ,'fleuprikrobla', 'greinklofloubra',
 'problifrouklebro','tireuchonkozein'
])
commands12 = [ 'doujusegani', 'fleublagropriklu' ,'fleuprikrobla', 'greinklofloubra',
 'problifrouklebro','tireuchonkozein']



model2 = load_model('model\model2.h5')

class_predictions2 = array([
    'coli', 'fabo', 'lifu'
])
commands2 = [ 'coli', 'fabo', 'lifu']

model3 = load_model('model\model3.h5')

class_predictions3 = array([
    'fege', 'pa', 'rak', 'reu', 'vet', 'vu'
])
commands3 = ['fege', 'pa', 'rak', 'reu', 'vet', 'vu']


model4 = load_model('model\model4.h5')

class_predictions4 = array([
   'coc' ,'ga' ,'ni', 'pan' ,'tab', 'zin'
])
commands4 = [  'coc' ,'ga' ,'ni', 'pan' ,'tab', 'zin']


model5 = load_model('model\model5.h5')

class_predictions5 = array([
 'au', 'chou', 'cre', 'doui', 'jou', 'pi', 'pri', 'ta', 'tan', 'train', 'ture', 'un'
 'voi', 'yon'
])
commands5 = [ 'au', 'chou', 'cre', 'doui', 'jou', 'pi', 'pri', 'ta', 'tan', 'train', 'ture', 'un'
 'voi', 'yon']



model6 = load_model('model\model6.h5')

class_predictions6 = array([
 'ba', 'dou', 'fro', 'ic', 'ou', 'us'
])
commands6 = ['ba', 'dou', 'fro', 'ic', 'ou', 'us' ]



model7 = load_model('model\model7.h5')

class_predictions7 = array([
 'br', 'jon', 'lei', 'lif', 'oui', 'pa'
])
commands7 = [ 'br', 'jon', 'lei', 'lif', 'oui', 'pa']



model8 = load_model('model\model8.h5')

class_predictions8 = array([
 'an', 'b', 'c', 'f', 'ou', 't'
])
commands8 = [ 'an', 'b', 'c', 'f', 'ou', 't']



model9 = load_model('model\model9.h5')

class_predictions9 = array([
 'anv', 'egg', 'rasle', 'ron', 'vion', 'wa'
])
commands9 = ['anv', 'egg', 'rasle', 'ron', 'vion', 'wa' ]




model10 = load_model('model\model10.h5')

class_predictions10 = array([
 'gamb', 'imeub', 'li', 'meni', 'oghe', 'tibu'
])
commands10 = ['gamb', 'imeub', 'li', 'meni', 'oghe', 'tibu' ]



model11 = load_model('model\model11.h5')

class_predictions11 = array([
 'chein', 'ib', 'ich', 'lo', 'oq', 'zen'
])
commands11 = [ 'chein', 'ib', 'ich', 'lo', 'oq', 'zen']

def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


AUTOTUNE = tf.data.AUTOTUNE


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.math.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return output_ds


DATASET_PATH = 'files'
data_dir = pathlib.Path(DATASET_PATH)


@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Recognition API!"}


@app.post("/net/voice/prediction1/")
async def get_net_audio_prediction1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands1
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model1(spectrogram)

    score = softmax(pred[0])

    class_prediction1 = class_predictions1[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return class_prediction1
        


@app.post("/net/voice/prediction12/")
async def get_net_audio_prediction12(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands12
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model1(spectrogram)

    score = softmax(pred[0])

    class_prediction12 = class_predictions12[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction12,
        "model-prediction-confidence-score": model_score
    }


@app.post("/net/voice/prediction2/")
async def get_net_audio_prediction2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands2
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model2(spectrogram)

    score = softmax(pred[0])

    class_prediction2 = class_predictions2[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction2,
        "model-prediction-confidence-score": model_score
    }

@app.post("/net/voice/prediction3/")
async def get_net_audio_prediction3(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands3
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model3(spectrogram)

    score = softmax(pred[0])

    class_prediction3 = class_predictions3[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction3,
        "model-prediction-confidence-score": model_score
    }

@app.post("/net/voice/prediction4/")
async def get_net_audio_prediction4

(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands4
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model4(spectrogram)

    score = softmax(pred[0])

    class_prediction4 = class_predictions4[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction4,
        "model-prediction-confidence-score": model_score
    }

@app.post("/net/voice/prediction5/")
async def get_net_audio_prediction5(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands5
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model5(spectrogram)

    score = softmax(pred[0])

    class_prediction5 = class_predictions5[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction5,
        "model-prediction-confidence-score": model_score
    }


@app.post("/net/voice/prediction6/")
async def get_net_audio_prediction6(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands6
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model6(spectrogram)

    score = softmax(pred[0])

    class_prediction6 = class_predictions6[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction6,
        "model-prediction-confidence-score": model_score
    }

@app.post("/net/voice/prediction7")
async def get_net_audio_prediction7(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands7
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model7(spectrogram)

    score = softmax(pred[0])

    class_prediction7 = class_predictions7[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction7,
        "model-prediction-confidence-score": model_score
    }



@app.post("/net/voice/prediction8/")
async def get_net_audio_prediction8(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands8
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model8(spectrogram)

    score = softmax(pred[0])

    class_prediction8 = class_predictions8[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction8,
        "model-prediction-confidence-score": model_score
    }


@app.post("/net/voice/prediction9/")
async def get_net_audio_prediction9(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands9
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model9(spectrogram)

    score = softmax(pred[0])

    class_prediction9 = class_predictions9[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction9,
        "model-prediction-confidence-score": model_score
    }



@app.post("/net/voice/prediction10/")
async def get_net_audio_prediction10(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands10
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model10(spectrogram)

    score = softmax(pred[0])

    class_prediction10 = class_predictions10[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction10,
        "model-prediction-confidence-score": model_score
    }



@app.post("/net/voice/prediction11/")
async def get_net_audio_prediction11(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    global commands
    commands = commands11
    sample_file = data_dir/f"{uploaded_file.filename}"
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        pred = model11(spectrogram)

    score = softmax(pred[0])

    class_prediction11 = class_predictions11[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction11,
        "model-prediction-confidence-score": model_score
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
