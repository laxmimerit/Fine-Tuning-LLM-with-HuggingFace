import streamlit as st
import pandas as pd
import numpy as np

from transformers import pipeline

st.title('Fine Tuning BERT for Sentiment Multi Class Sentiment Classification')

pipe = pipeline("text-classification", model="bert-base-uncased-sentiment-model")

text = st.text_area('Enter some text')

if st.button('Predict'):
    result = pipe(text)
    st.write(result)

