from PIL import Image
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import tkinter as tk
from tkinter import filedialog
import glob
import SessionState as SS

stroke_width=3
stroke_color = "#000000"
bg_color = "#eee"

# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

btn_dbdir = st.button('Select new dataset directory')
if btn_dbdir:
    dir_db = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

btn_inputdir = st.button('Select input image directory')
ss = SS.get(file_list='',counter=0, output_dir='')


if btn_inputdir:
    dir_in = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    ss.file_list = glob.glob(dir_in+"/*.jpg")


btn_next = st.button('Next')

if btn_next:
    ss.counter += 1
print(ss.file_list[ss.counter])
bg_image = cv2.imread(ss.file_list[ss.counter])

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(ss.file_list[ss.counter]),
    update_streamlit=True,
    height=451,
    drawing_mode='rect',
    initial_drawing=None,
    display_toolbar=True,
    key="canvas0",
)

# st.image(img)
    # print('aa '+dir_in)
    # file_list = glob.glob(dir_in + "/*.jpg")
    # print(file_list[0])

    # path = file_list[co]
    # print(path)


