import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from PIL import Image
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l1,l2

def mnist_func():
    st.title("MNIST")
    st.write("This dataset is a collection of Handwritten Digit from 0-9. This Dataset is used for training a model that can classify or recognize handwritten digits from an image.")

    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    classes = [0,1,2,3,4,5,6,7,8,9]

    st.write(" ")
    st.markdown('**Shape**')
    st.write('\nTraining dataset :',x_train.shape, "\nTesting dataset :",x_test.shape)

    st.write("**Data** ")

    rand_14 = np.random.randint(0, x_train.shape[0],14)
    sample_digits = x_train[rand_14]
    sample_labels = y_train[rand_14]

    num_rows, num_cols = 2, 7
    f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
                     gridspec_kw={'wspace':0.03, 'hspace':0.01},
                     squeeze=True)

    for r in range(num_rows):
        for c in range(num_cols):
            image_index = r * 7 + c
            ax[r,c].axis("off")
            ax[r,c].imshow(sample_digits[image_index], cmap='gray')
            ax[r,c].set_title('%s' % classes[int(sample_labels[image_index])])
    plt.show()
    st.pyplot(clear_figure=False)
    plt.close()

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

    image_height = x_train.shape[1]
    image_width = x_train.shape[2]
    num_channels = 1

    x_train = np.reshape(x_train, (x_train.shape[0], image_height, image_width, num_channels))
    x_test = np.reshape(x_test, (x_test.shape[0],image_height, image_width, num_channels))

    x_train,x_test = x_train/255.0,x_test/255.0

    y_train,y_test = to_categorical(y_train),to_categorical(y_test)


    st.write("")
    st.write("**Build Model**")

    act = st.selectbox( "Choose the type of Activation Function ",('relu','sigmoid', 'tanh'))

    pad = st.selectbox("Choose the Padding ",('same','valid'))

    dropout = st.checkbox("Dropout")


    opt = st.selectbox("Choose the type of Optimizer ",("adam","sgd","rmsprop","adagrad"))

    val = st.checkbox('Validation Set')
    epoch = st.slider("Epochs",0,250,step=1)
    b_s = st.slider("Batch Size",32,1024,step=32)

    st.write("")
    st.write("")


    if st.button("Train Model"):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation=act, padding=pad,input_shape=(image_height, image_width, num_channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation=act, padding=pad))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation=act, padding=pad))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation=act, padding=pad))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation=act))
        model.add(Dense(len(classes), activation='softmax'))
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
            if val:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch),validation_split=0.2)
            else:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch))
        st.success("Model Trained.")
        results = model.evaluate(x_test,y_test,batch_size=128)
        st.write("Loss: ",results[0])
        st.write("Accuracy: ",results[1])

        st.write("**Predictions** (Random Test Samples)")
        Images = []
        pred = ""

        for i in range(5):
            r = np.random.randint(0,len(x_test))
            Images.append(x_test[r].reshape(x_train.shape[1],x_train.shape[2]))
            pred += str(classes[model.predict(x_test[r].reshape(-1,x_train.shape[1],x_train.shape[2],1)).argmax()])
            pred += " "

        st.image(Images,width=100)
        st.write(pred)
