import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from detect_mask_video import processFrame
from detect_mask_video import loadModels
import imutils
import pandas as pd

st.title("😷 Face Mask Detection 😷")
happy_url = "smiley/happy.png"
put_url = "smiley/angry.png"
incorrect_url = "smiley/not-happy.png"
handle_image = st.image(happy_url, width=400)
col1, col2 = st.columns(2)

cols = ['incorrect', 'mask', 'nomask']
timeseries = np.array([[0, 0, 1], [0, 0, 0]])

run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

chart_data = pd.DataFrame(timeseries, columns=cols)
handle = st.area_chart(data = chart_data, use_container_width = True)

camera = cv2.VideoCapture(0)
(faceModel, maskModel) = loadModels()

while run:
    _, frame = camera.read()
    (incorrect, mask, nomask) = processFrame(frame, faceModel, maskModel)
    if(nomask == 0 and incorrect == 0 and mask > 0):
        handle_image.image(happy_url, width=400)
    
    elif(nomask > 0):
        handle_image.image(put_url, width=400)
    
    elif(incorrect > 0):
        handle_image.image(incorrect_url, width=400)
    
    timeseries = np.append(timeseries, [[incorrect, mask, nomask]], axis=0)
    chart_data = pd.DataFrame(timeseries, columns=cols)
    handle.area_chart(chart_data)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')