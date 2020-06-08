import streamlit as st
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.datasets import imdb

def imdb_func():
    st.title("IMDB")
    st.write("The dataset is a collection of highly polar movie reviews from IMDB. This dataset is used for training a binary sentiment classification model.")

    max_features = 20000
    with st.spinner('Loading Data...'):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

    st.write(" ")
    st.markdown('**Shape**')
    st.write('\nTraining dataset :',x_train.shape, "\nTesting dataset :",x_test.shape)


    classes = ["Negative", "Positive"]


    st.write(" ")
    st.write("**Data** ")
    index = imdb.get_word_index()
    feature = []
    for i in range(3):
        reverse_index = dict([(value, key) for (key, value) in index.items()])
        decoded = " ".join( [reverse_index.get(j - 3, "#") for j in x_train[i]])
        decoded = decoded[1:]
        feature.append(decoded)
    label = [classes[y_train[i]] for i in range(3)]

    df = pd.DataFrame(list(zip(feature,label)),columns = ['Reviews','Sentiments'])
    st.table(df)


    st.write("**Classes** ")
    s = ""
    for i in range (len(classes)):
        if i is not (len(classes)-1):
            s += str(classes[i]).title()
            s += ","
            s += " "
        else:
            s += str(classes[i])
    st.write(s)

    x_train = sequence.pad_sequences(x_train, maxlen=100)
    x_test = sequence.pad_sequences(x_test, maxlen=100)


    st.write("")
    st.write("**Build Model**")
    dropout = st.checkbox("Dropout")


    opt = st.selectbox("Choose the type of Optimizer ",("adam","sgd","rmsprop","adagrad"))

    val = st.checkbox('Validation Set')
    epoch = st.slider("Epochs",0,250,step=1)
    b_s = st.slider("Batch Size",32,1024,step=32)

    st.write("")
    st.write("")
    if st.button("Train Model"):
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, recurrent_dropout=0.2))
        if(dropout):
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
            if val:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch),validation_split=0.2)
            else:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch))
        st.success("Model Trained.")
        results = model.evaluate(x_test,y_test,batch_size=128)
        st.write("Loss: ",results[0])
        st.write("Accuracy: ",results[1])
        st.write("")
        st.write("**Predictions** (Random Test Samples)")

        l = []
        f = []

        index = imdb.get_word_index()
        for i in range(3):
            r = np.random.randint(0,len(x_test))
            reverse_index = dict([(value, key) for (key, value) in index.items()])
            decoded = " ".join( [reverse_index.get(j - 3, "#") for j in x_test[r]])
            decoded = decoded[1:]
            f.append(decoded)
            l.append(classes[y_train[r]])
        df2 = pd.DataFrame(list(zip(f,l)),columns = ['Reviews','Sentiments'])
        st.table(df2)
