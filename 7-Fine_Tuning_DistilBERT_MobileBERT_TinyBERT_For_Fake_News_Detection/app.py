# https://www.youtube.com/watch?v=hff2tHUzxJM&list=PLc2rvfiptPSSpZ99EnJbH5LjTJ_nOoSWW

import streamlit as st
import pandas as pd
import numpy as np

from transformers import pipeline

st.title("Fine Tuning BERT for Twitter Tweets for Multi Class Sentiment Classification")

classifier = pipeline('text-classification', model= 'bert-base-uncased-sentiment-model')

text = st.text_area("Enter some text")

if st.button("Predict"):
    result = classifier(text)
    st.write(result)