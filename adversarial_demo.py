#!/usr/bin/env python

import streamlit as st
from ttictoc import TicToc

import pickle
import lightgbm as lgb
import numpy as np

from art.attacks import ZooAttack
from art.classifiers import LightGBMClassifier
from art.utils import load_mnist

@st.cache
def load_data():
  return load_mnist(), pickle.load(open('data/adv_examples.pkl', 'rb'))

st.title("Adversarial Input Perturbation Generation Demo")
((x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value), x_test_adv = load_data()
# x_train.shape, y_train.shape, x_test.shape, y_test.shape, min_pixel_value, max_pixel_value
x_test.shape, x_test_adv.shape

st.image(x_test[4], width=280)
st.image(x_test_adv[4].reshape(28, 28), width=280)


# st.image(x_test[0], width=280)
# st.image(x_test_adv[0], width=280)



# st.markdown("Welcome! This is a simple NLP application created using Streamlit and deployed on Heroku.")
# st.markdown("In the box below, you can type custom text or paste an URL from which text is extracted. Once you have a the text, open the sidebar and choose any of the four applications. Currently, we have applications to tokenize text, extract entitiles, analyze sentiment, and summarize text (and a suprise! :wink:).")
# st.markdown("You can preview a percentage of your text by selecting a value on the slider and clicking on \"Preview\"")

# nlp = load_spacy()

# text = fetch_text(st.text_area("Enter Text (or URL) and select application from sidebar", "Here is some sample text. When inputing your custom text or URL make sure you delete this text!"))

# pct = st.slider("Preview length (%)", 0, 100)
# length = (len(text) * pct)//100
# preview_text = text[:length]

# if st.button("Preview"):
#   st.write(preview_text)

# apps = ['Show tokens & lemmas', 'Extract Entities', 'Show sentiment', 'Summarize text', 'Suprise']
# choice = st.sidebar.selectbox("Select Application", apps)
# if choice == "Show tokens & lemmas":
#   if st.button("Tokenize"):
#     st.info("Using spaCy for tokenization and lemmatization")
#     st.json([(f"Token: {token.text}, Lemma: {token.lemma_}") for token in analyze_text(nlp, text)])
# elif choice == 'Extract Entities':
#   if st.button("Extract"):
#     st.info("Using spaCy for NER")      
#     doc = analyze_text(nlp, text)
#     html = displacy.render(doc, style='ent')
#     html = html.replace('\n\n', '\n')
#     st.write(html, unsafe_allow_html=True)
# elif choice == "Show sentiment":
#   if st.button("Analyze"):
#     st.info("Using TextBlob for sentiment analysis")
#     blob = TextBlob(text)
#     sentiment = {
#       'polarity': np.round(blob.sentiment[0], 3),
#       'subjectivity': np.round(blob.sentiment[1], 3),
#     }
#     st.write(sentiment)
#     st.info("Polarity is between -1 (negative) and 1 (positive) indicating the type of sentiment\nSubjectivity is between 0 (objective) and 1 (subjective) indicating the bias of the sentiment")
# elif choice == "Summarize text":    
#   summarizer_type = st.sidebar.selectbox("Select Summarizer", ['Gensim', 'Sumy Lex Rank'])
#   if summarizer_type == 'Gensim':
#     summarizer = gensim_summarizer
#   elif summarizer_type == 'Sumy Lex Rank':
#     summarizer = sumy_summarizer

#   if st.button(f"Summarize using {summarizer_type}"):
#     st.success(summarizer(text))
# elif choice == 'Suprise':
#   st.balloons()

# st.markdown("The code for this app can be found in [this](https://github.com/sudarshan85/streamlit_nlp) Github repository.")    
