import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from detect_mask_video import processFrame
from detect_mask_video import loadModels
import imutils
import pandas as pd

st.title("Facemask Detector")

cols = ['incorrect', 'mask', 'nomask']
timeseries = np.array([[0, 0, 1], [0, 0, 0]])
chart_data = pd.DataFrame(timeseries, columns=cols)
handle = st.area_chart(data = chart_data, height = 2, use_container_width = True)

run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
(faceModel, maskModel) = loadModels()

while run:
    _, frame = camera.read()
    (incorrect, mask, nomask) = processFrame(frame, faceModel, maskModel)
    timeseries = np.append(timeseries, [[incorrect, mask, nomask]], axis=0)
    chart_data = pd.DataFrame(timeseries, columns=cols)
    handle.area_chart(chart_data)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')