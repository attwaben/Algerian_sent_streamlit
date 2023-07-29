import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from keras import layers
#print('hello world')

st.set_page_config(page_title='my_app', layout="wide")

with st.container():

    st.header('this is my first app with streamlit')

    st.subheader('Try this app and send me your feedback')

    st.write('This is an app that takes an Algerian arabic sentence then returns its sentiment')


text = st.text_area('text to analyse  <press Ctrl+Enter  to apply>')
print(text)

def tokenize_sentence(sentence, vocab):
    # Tokenize the sentence into individual words or subwords
    tokens = sentence.split()

    # Map tokens to their corresponding indices in the vocabulary

    input_ids = [vocab[token] for token in tokens if token in vocab ]

    return input_ids

def pad_or_truncate(input_ids, max_length, padding_token_id):
    # Pad or truncate the input_ids list to the desired max_length
    if len(input_ids) < max_length:
        input_ids = input_ids + [padding_token_id] * (max_length - len(input_ids))
    else:
        input_ids = input_ids[:max_length]

    return input_ids


vocab2= pd.read_csv("tokens.csv")

dict_ = dict(zip(vocab2['word'], vocab2['rank']))

#print(dict_)
#test = "اجمل اعلى أنا احب الجزائر الله اكبر"
# Preprocess the sentence
input_ids = tokenize_sentence(text, dict_)
input_ids = pad_or_truncate(input_ids, max_length=25, padding_token_id=0)
input_ids = tf.reshape(input_ids,[1,25])

print(input_ids)



#test_ds = vectorize_text(input_ids)

#print(input_ids)

model = tf.keras.models.load_model('algerian_sent_model.h5')



predictions = model.predict(input_ids)

st.write('Sentiment')

#def sent() :

    #if sum(input_ids[1:])==0:
        #st.success('More data is needed')
    #elif np.argmax(predictions) > 0.5 :
        #st.success('your sentence is  Positive' )
    #else :
        #st.success('your sentence is  Negative' )

#print (a)
#sent()
sent_=[]
sent_=[ 'Positive' if np.argmax(predictions) > 0.5 else 'Negative']
np.argmax(predictions)

#print(a)
#print(sent_)
#sent_=['More data is needed', 'Positive', 'Negative']

st.success('your sentence is  ' + str(sent_[0]))

#if sum(input_ids[1:])==0 :
    #st.write('More data is needed')
#else :
    #pass
