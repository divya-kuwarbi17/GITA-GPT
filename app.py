import streamlit as st
import pandas as pd
from model_v1 import get_ans
import time

st.write("""
# Welcome to gita gpt,
if you have any question, ask here!!!
""")

query = st.text_input('Enter your Question: ', "Name the pandu's son?")
answer = get_ans(query)
text = "Welcome to the first day... of the rest... of your life"


t = st.empty()
for i in range(len(answer) + 1):
    t.markdown("## %s..." % text[0:i])
    time.sleep(0.1)

