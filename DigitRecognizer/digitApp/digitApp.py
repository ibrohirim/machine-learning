import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from keras.models import load_model
from tensorflow import convert_to_tensor
import tensorflow as tf
import pandas as pd
import math
import matplotlib.pyplot as plt

from PIL import Image


def main():
    def convertImage(image):
        if image is not None:
            image = Image.fromarray(np.uint8(image)).convert('L')
            image = np.array(image)
            while np.sum(image[0]) == 0:
                image = image[1:]
            while np.sum(image[:,0]) == 0:
                image = np.delete(image, 0, 1)
            while np.sum(image[-1]) == 0:
                image = image[:-1]
            while np.sum(image[:, -1]) == 0:
                image = np.delete(image, -1, 1)
            rows, cols = image.shape
            colsPadding = (int(math.ceil((308-cols)/2.0)), int(math.floor((308-cols)/2.0)))
            rowsPadding = (int(math.ceil((308-rows)/2.0)), int(math.floor((308-rows)/2.0)))
            image = np.lib.pad(image, (rowsPadding, colsPadding), 'constant')
            image = Image.fromarray(np.uint8(image))
            image = image.resize(size=(28, 28))
            st.image(image)
            image = np.array(image)
            image = image.reshape(1, 28, 28, 1)
            image = tf.convert_to_tensor(image)
        return image

    def loadModel():
        model = load_model('./saved_model.h5', compile=True)
        return model

    def prediction(m, d):
        pred = m(d, training=False)
        pred = pred.numpy()
        return pred


    # Load the model
    model = loadModel()


    st.write("""
    #My first app
    Hello World
    """)
    # Specify brush parameters and drawing mode
    drawing_mode = st.sidebar.selectbox("Drawing mode", ("freedraw", "line", "transform"))
    key  = "canvas"
    # Create a canvas component
    image_data = st_canvas(
            20,
            "#FFF",
            "#000",
            height=308,
            width=308,
            drawing_mode=drawing_mode,
            key = key,
        )



    if image_data is not None:
        if np.mean(image_data) != 63.75:
            data  = convertImage(image_data)
            pred = prediction(model, data)
            print(pred)
            xt = ('0', '1', '2', '3', '4', '5', '6', '7', '8','9')
            y_pos = np.arange(len(xt))
            plt.bar(y_pos, height=pred[0], align='center', color=['blue', 'red', 'orange', 'green', 'black', 'cyan', 'yellow', 'purple', 'magenta', 'pink'])
            plt.xticks(y_pos, xt)
            st.pyplot(plt)


if __name__ == "__main__":
    main()