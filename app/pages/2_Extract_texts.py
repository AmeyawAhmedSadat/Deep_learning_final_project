#!/usr/bin/env python
# coding: utf-8

import os
import io
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import pipeline

# Page headers and descriptions (must be the first Streamlit command)
st.set_page_config(
    page_title='OCR and Text Summarization', 
    page_icon=':page_facing_up:'
)
st.sidebar.header('OCR and Summarization')
st.header('OCR and Text Summarization Tool', divider='rainbow')

st.markdown(
    """
    This application allows you to extract text from uploaded images or PDFs.
    Additionally, you can generate a concise summary of the extracted text.
    
    **Steps:**
    1. Upload an image or PDF document.
    2. Select the language for text extraction.
    3. Extract text using OCR.
    4. Optionally, generate a summary of the extracted text.
    """
)
st.divider()

####################################
########## YOUR CODE HERE ##########
####################################
# Implementing the text summarization model
with st.spinner('Please wait, application is initializing...'):
    MODEL_SUM_NAME = 'sshleifer/distilbart-cnn-12-6'
    SUMMARIZATOR = pipeline("summarization", model=MODEL_SUM_NAME)
####################################

# File upload
uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
lang = st.selectbox(
    'Select language to extract text from:',
    ('eng', 'rus', 'eng+rus')
)

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Convert PDF pages to images
        pdf_pages = convert_from_bytes(uploaded_file.read())
        st.write("PDF contains {} pages.".format(len(pdf_pages)))
        
        # Display and process each page
        full_text = ""
        for page_num, page in enumerate(pdf_pages, start=1):
            st.image(page, caption=f'Page {page_num}', use_column_width=True)
            page_text = pytesseract.image_to_string(page, lang=lang)
            full_text += page_text
            st.text_area(f'Extracted Text - Page {page_num}', page_text, height=200)
    
    else:
        # Process image files
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        full_text = pytesseract.image_to_string(image, lang=lang)
        st.text_area('Extracted Text', full_text, height=200)

    # Summarization option
    st.divider()
    if st.button("Generate Summary"):
        if full_text:
            with st.spinner('Generating summary...'):
                summary = SUMMARIZATOR(full_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                st.subheader("Summary")
                st.write(summary)
        else:
            st.error("No text found to summarize.")

st.divider()
st.markdown("Made with ❤️ using Tesseract OCR and Hugging Face Transformers by Sadat Ahmed")
