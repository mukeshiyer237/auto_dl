import streamlit as st
from mnist import *
from cifar10 import *
from cifar100 import *
from fashion_mnist import *
from imdb import *

if __name__ == "__main__":
    st.sidebar.header("AutoDL")
    st.sidebar.markdown("Deep Learning Web App built using Streamlit")
    st.sidebar.markdown(" ")
    dataset = st.sidebar.selectbox("Select the dataset that you wish to work on ",('MNIST', 'Fashion_MNIST','CIFAR-10', 'CIFAR-100','IMDB'))
    if dataset == "MNIST":
        mnist_func()
    elif dataset == "Fashion_MNIST":
        fashion_mnist_func()
    elif dataset == "CIFAR-10":
        cifar10_func()
    elif dataset == "CIFAR-100":
        cifar100_func()
    elif dataset == "IMDB":
        imdb_func()
