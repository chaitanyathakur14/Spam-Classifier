import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')
def preprocess(text: str):
    # Converting to lowercase
    text = text.lower()

    # Tokenization
    textToken = []
    textToken = nltk.word_tokenize(text,)

    # Removing special characters
    alphanum = []
    for word in textToken:
        if word.isalnum():
            alphanum.append(word)

    # stemming
    stemlist = []
    ps = PorterStemmer()
    for word in alphanum:
        stemlist.append(ps.stem(word))

    # Removing stop words
    nonStopList = []
    for word in stemlist:
        if word not in stopwords.words('english') and word not in string.punctuation:
            nonStopList.append(word)
    return " ".join(nonStopList)


tfidf = pickle.load(open('assets/vectorizer.pkl', 'rb'))
model = pickle.load(open('assets/model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = preprocess(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
