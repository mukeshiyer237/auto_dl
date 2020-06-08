import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import MaxPool2D, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import tensorflow as  tf
import matplotlib.pyplot as plt

def cifar100_func():
    st.title("CIFAR-100")
    st.write("The dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. This dataset is used for training a multiclass classification model that can classify or recognize images belonging to 100 different classes.")

    from keras.datasets import cifar100

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    classes = ['beaver', ' dolphin', ' otter', ' seal', ' whale', ' aquarium fish', ' flatfish', ' ray', ' shark', ' trout', ' orchids', ' poppies', ' roses', ' sunflowers', ' tulips', ' bottles', ' bowls', ' cans', ' cups', ' plates', ' apples', ' mushrooms', ' oranges', ' pears', ' sweet peppers', ' clock', ' computer keyboard', ' lamp', ' telephone', ' television', ' bed', ' chair', ' couch', ' table', ' wardrobe', ' bee', ' beetle', ' butterfly', ' caterpillar', ' cockroach', ' bear', ' leopard', ' lion', ' tiger', ' wolf', ' bridge', ' castle', ' house', ' road', ' skyscraper', ' cloud', ' forest', ' mountain', ' plain', ' sea', ' camel', ' cattle', ' chimpanzee', ' elephant', ' kangaroo', ' fox', ' porcupine', ' possum', ' raccoon', ' skunk', ' crab', ' lobster', ' snail', ' spider', ' worm', ' baby', ' boy', ' girl', ' man', ' woman', ' crocodile', ' dinosaur', ' lizard', ' snake', ' turtle', ' hamster', ' mouse', ' rabbit', ' shrew', ' squirrel', ' maple', ' oak', ' palm', ' pine', ' willow', ' bicycle', ' bus', ' motorcycle', ' pickup truck', ' train', ' lawn-mower', ' rocket', ' streetcar', ' tank', ' tractor']

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
    num_channels = 3


    x_train_min = x_train.min(axis=(1, 2), keepdims=True)
    x_train_max = x_train.max(axis=(1, 2), keepdims=True)
    x_train = (x_train - x_train_min)/(x_train_max-x_train_min)

    x_test_min = x_test.min(axis=(1, 2), keepdims=True)
    x_test_max = x_test.max(axis=(1, 2), keepdims=True)
    x_test = (x_test - x_test_min)/(x_test_max-x_test_min)


    x_train = np.reshape(x_train, (x_train.shape[0], image_height, image_width, num_channels))
    x_test = np.reshape(x_test, (x_test.shape[0],image_height, image_width, num_channels))


    y_train,y_test = to_categorical(y_train),to_categorical(y_test)


    st.write("")
    st.write("**Build Model**")

    act = st.selectbox( "Choose the type of Activation Function ",('elu','relu','sigmoid','nadam', 'tanh'))

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

        model.add(Conv2D(128, (3, 3), padding=pad, activation=act, input_shape=(32, 32, 3)))
        model.add(Conv2D(128, (3, 3), activation=act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            model.add(Dropout(0.1))

        model.add(Conv2D(256, (3, 3), padding=pad, activation=act))
        model.add(Conv2D(256, (3, 3), activation= act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding=pad, activation=act))
        model.add(Conv2D(512, (3, 3), activation=act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation=act))
        model.add(Dropout(0.5))
        model.add(Dense(len(classes), activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
            if val:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch),validation_split=0.2)
            else:
                result = model.fit(x_train,y_train,batch_size=int(b_s),epochs=int(epoch))

        st.success("Model Trained.")
        results = model.evaluate(x_test,y_test,batch_size=128)
        st.write("Loss: ",results[0])
        st.write("Accuracy: ",results[1])
        model.save("model/cifar100.h5")
        st.write("**Predictions** (Random Test Samples)")
        Images = []
        pred = ""

        for i in range(5):
            r = np.random.randint(0,len(x_test))
            Images.append(x_test[r].reshape(x_train.shape[1],x_train.shape[2]))
            pred += str(classes[model.predict(x_test[r].reshape(-1,x_train.shape[1],x_train.shape[2],3)).argmax()])
            pred += " "

        st.image(Images,width=100)
        st.write(pred)
