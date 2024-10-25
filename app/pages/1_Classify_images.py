#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from deepface import DeepFace
from transformers import pipeline
import pandas as pd
import numpy as np

def img_caption(model, processor, img, text=None):
    """
    Uses BLIP model to caption image.
    """
    res = None
    if text:
        # Conditional image captioning
        inputs = processor(img, text, return_tensors='pt')
    else:
        # Unconditional image captioning
        inputs = processor(img, return_tensors='pt')
    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)
    return res

# Change YOLO detection to segmentation model (using YOLOv8 segmentation because 11 doesn't work)
def img_detect(model, img, plot=False):
    """
    Run YOLO segmentation on an image.
    """
    result = model(img)[0]
    masks = result.masks  # Segmentation masks

    # If masks are None, handle the case
    if masks is None:
        st.error("No segmentation masks found.")
        return None, None

    img_bgr = result.plot()
    img_rgb = Image.fromarray(img_bgr[..., ::-1])  # Convert to RGB

    if plot:
        plt.figure(figsize=(16, 8))
        plt.imshow(img_rgb)
        plt.show()

    return masks, img_rgb

def zeroshot(classifier, classes, img):
    scores = classifier(img, candidate_labels=classes)
    return scores

def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data

# Page headers and info text
st.set_page_config(page_title='Classify images', page_icon=':bar_chart:')
st.sidebar.header('Classify images')
st.header('AI-assistant for your images', divider='rainbow')
st.markdown(
    """
    You can upload your image here and it will be 
    classified by predefined classes and saved to
    gallery.
    The assistant can also give a caption to the image
    and find your friends at the photo.
    """
)
st.divider()

# Uploading models
with st.spinner('Please wait, application is initializing...'):
    # Caption model
    MODEL_CAP_NAME = 'Salesforce/blip-image-captioning-base'
    PROCESSOR_CAP = BlipProcessor.from_pretrained(MODEL_CAP_NAME)
    MODEL_CAP = BlipForConditionalGeneration.from_pretrained(MODEL_CAP_NAME)

    # Detection objects model
    MODEL_DET_NAME = 'yolov8n.pt'
    MODEL_DET = YOLO(MODEL_DET_NAME)
    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # Experiment with new version YOLO
    # models and try segmentation model
    # as an alternative:
    # https://docs.ultralytics.com/tasks/segment/
    MODEL_SEG_NAME = 'yolov8n-seg.pt'
    MODEL_SEG = YOLO(MODEL_SEG_NAME)
    # Use YOLOv8 for segmentation

    ####################################

    # Classification model and classes
    MODEL_ZERO_NAME = 'openai/clip-vit-base-patch16'
    CLASSIFIER_ZERO = pipeline('zero-shot-image-classification', model=MODEL_ZERO_NAME)
    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # You may want to change the list of
    # classes so you will have to edit
    # `config.yaml` file and update it
    # with new data.
    ####################################
    APP_CONFIG = read_json(file_path='config.json')
    CLASSES = APP_CONFIG['classes']
    DB_DICT = APP_CONFIG['db_dict']
    TH_OTHERS = APP_CONFIG['th_others']
    IMGS_PATH = APP_CONFIG['imgs_path']

    # Create directories for each category
    for k, v in DB_DICT.items():
        imgs_path = f'{IMGS_PATH}/{v.strip()}'
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)
    imgs_path = f'{IMGS_PATH}/other'
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # Faces detection and recognition config
    DEEPFACE_MODELS = [
        'VGG-Face', 
		'Facenet', 
		'Facenet512', 
		'OpenFace', 
        'DeepFace', 
		'DeepID', 
		'ArcFace', 
		'Dlib', 
        'SFace', 'GhostFaceNet'
    ]
    DB_PATH = '/home/jovyan/Deep_Learning/dlba_course_miba_24/topic_09/app/data/db'

# Image upload section
st.write('#### Upload your image')
uploaded_file = st.file_uploader('Select an image file (JPEG format)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    if '.jpg' in file_name:
        # Input text for conditional image captioning
        text = st.text_input('Input text for conditional image captioning (if needed)', '')
        with st.spinner('Please wait...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))

            # Image caption model for uploaded image
            caption = img_caption(model=MODEL_CAP, processor=PROCESSOR_CAP, img=img, text=text)
            st.write('##### Your image uploaded')
            st.image(img, caption=caption)
            st.divider()
            # Logging
            msg = '{} - file "{}" got caption "{}"\n'.format(datetime.datetime.now(), file_name, caption)
            with open('history.log', 'a') as file:
                file.write(msg)

            # Object segmentation model for uploaded image
            st.write('##### Objects detected')
            masks, img_det = img_detect(model=MODEL_SEG, img=img)

            # Display masks if found
            if masks is not None:
                st.image(img_det, caption='Segmentation results', width=800)
                st.divider()
                st.caption('Segmentation masks found:')
            else:
                st.write("No objects detected in segmentation.")

            # Logging segmentation results
            msg = '{} - file "{}" objects detected\n'.format(datetime.datetime.now(), file_name)
            with open('history.log', 'a') as file:
                file.write(msg)

            # Zero-shot image classification
            scores = zeroshot(classifier=CLASSIFIER_ZERO, classes=CLASSES, img=img)
            max_score = sorted(scores, key=lambda x: x['score'])[-1]
            category = max_score['label'] if max_score['score'] >= TH_OTHERS else 'a photo of unknown stuff'
            save_path = DB_DICT.get(category, 'other')
            img.save(f'{IMGS_PATH}/{save_path}/{file_name}')
            st.write('##### Classification results')
            st.image(img, caption=f'Looks like it is {category}')
            st.write(f'Image saved as: {save_path}')
            st.divider()

            # Logging classification results
            msg = '{} - file "{}" saved as "{}" category "{}"\n'.format(datetime.datetime.now(), file_name, save_path, category)
            with open('history.log', 'a') as file:
                file.write(msg)

            # Plot classification scores
            df = pd.DataFrame(scores).set_index('label')
            st.bar_chart(df)
            st.divider()

            # Faces detection and recognition
            results = DeepFace.find(
                img_path=np.array(img), db_path=DB_PATH, model_name=DEEPFACE_MODELS[0], enforce_detection=False
            )
            st.write('##### Faces recognition results')
            found = []
            for result in results:
                name = result.identity.values
                if isinstance(name, np.ndarray) and name.size > 0:
                    found.append(name[0].replace(f'{DB_PATH}/', '').replace('.jpg', ''))
                elif isinstance(name, list) and len(name) > 0:
                    found.append(name[0].replace(f'{DB_PATH}/', '').replace('.jpg', ''))
                elif isinstance(name, str) and name:
                    found.append(name.replace(f'{DB_PATH}/', '').replace('.jpg', ''))

            # If faces are found, perform emotion detection
            if found:
                st.write(f'Found: {" ,".join(found)}')
                for person in found:
                    face_img_path = f'{DB_PATH}/{person}.jpg'
                    analysis = DeepFace.analyze(face_img_path, actions=['emotion'])
                    dominant_emotion = analysis[0]['dominant_emotion']
                    st.write(f'{person} is feeling {dominant_emotion}')
            else:
                st.write('No known faces found')

            # Logging faces recognition results
            msg = '{} - file "{}" {}\n'.format(
                datetime.datetime.now(), file_name, 'faces found: ' + ', '.join(found) if found else 'no known faces found'
            )
            with open('history.log', 'a') as file:
                file.write(msg)
    else:
        st.error('File read error', icon='⚠️')
