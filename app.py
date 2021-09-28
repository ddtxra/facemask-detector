import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from detect_mask_video import processFrame
from detect_mask_video import loadModels
import imutils

st.title("Facemask Detector")

incorrect = 0
mask = 0
nomask = 0

status_text = st.empty()
status_text.text("Mask: " + str(mask) + "incorrect: " + str(incorrect) + "NoMask: " + str(nomask))

run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
(faceModel, maskModel) = loadModels()

while run:
    _, frame = camera.read()
    #frame = imutils.resize(frame, width=400)
    (i, m, n) = processFrame(frame, faceModel, maskModel)
    mask = mask + m
    incorrect = incorrect + i
    nomask = nomask + n

    status_text.text("Mask: " + str(mask) + "incorrect: " + str(incorrect) + "NoMask: " + str(nomask))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')