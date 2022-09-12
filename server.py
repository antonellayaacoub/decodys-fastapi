from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import pathlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from IPython import display
from numpy import argmax
from numpy import max
from numpy import array
from uvicorn import run
import os
from tflite_support import metadata
import json
import librosa

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


AUTOTUNE = tf.data.AUTOTUNE


def get_labels(model):
    """Returns a list of labels, extracted from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    labels_file = displayer.get_packed_associated_file_list()[0]
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return [line for line in labels.split('\n')]


def get_input_sample_rate(model):
    """Returns the model's expected sample rate, from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    metadata_json = json.loads(displayer.get_metadata_json())
    input_tensor_metadata = metadata_json['subgraph_metadata'][0][
        'input_tensor_metadata'][0]
    input_content_props = input_tensor_metadata['content']['content_properties']
    return input_content_props['sample_rate']


DATASET_PATH = 'files'
data_dir = pathlib.Path(DATASET_PATH)
SAVE_PATH = './models'


@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Recognition API!"}


@app.post("/net/voice/prediction1_1/")
async def get_net_audio_prediction1_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-1_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction1_2/")
async def get_net_audio_prediction1_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-1_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction1_3/")
async def get_net_audio_prediction1_3(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-1_3.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction1_4/")
async def get_net_audio_prediction1_4(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-1_4.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction1_5/")
async def get_net_audio_prediction1_5(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-1_5.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction2/")
async def get_net_audio_prediction2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction3_1/")
async def get_net_audio_prediction3_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-3_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction3_2/")
async def get_net_audio_prediction3_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-3_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction4_1/")
async def get_net_audio_prediction4_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-4_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction4_2/")
async def get_net_audio_prediction4_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-4_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction7_1/")
async def get_net_audio_prediction7_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-7_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction7_2/")
async def get_net_audio_prediction7_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-7_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction8_1/")
async def get_net_audio_prediction8_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-8_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction8_2/")
async def get_net_audio_prediction8_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-8_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction9_1/")
async def get_net_audio_prediction9_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-9_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction9_2/")
async def get_net_audio_prediction9_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-9_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction10_1/")
async def get_net_audio_prediction10_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-10_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction10_2/")
async def get_net_audio_prediction10_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-10_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction11_1/")
async def get_net_audio_prediction11_1(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-11_1.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


@app.post("/net/voice/prediction11_2/")
async def get_net_audio_prediction11_2(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    sample_file = data_dir/f"{uploaded_file.filename}"
    TFLITE_FILENAME = 'browserfft-speech-11_2.tflite'
    tflite_file = os.path.join(SAVE_PATH, TFLITE_FILENAME)
    labels = get_labels(tflite_file)

    interpreter = tf.lite.Interpreter(tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    sample_rate = get_input_sample_rate(tflite_file)
    audio_data, _ = librosa.load(sample_file, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {
        "label": label,
        "score": str(score)
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
