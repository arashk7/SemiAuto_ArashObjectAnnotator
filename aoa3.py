from PIL import Image
import cv2
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import tkinter as tk
from tkinter import filedialog
import glob
import SessionState as SS

stroke_width = 3
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
ss = SS.get(file_list='', counter=0, output_dir='', clear=False)

if btn_inputdir:
    dir_in = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    ss.file_list = glob.glob(dir_in + "/*.jpg")

btn_dr_mode = st.radio('Drawing mode:', ('rect', 'transform'))
# print(ss.file_list[ss.counter])
bg_image = cv2.imread(ss.file_list[ss.counter])
basename = os.path.basename(os.path.normpath(ss.file_list[ss.counter]))

file_name = basename.split('.')[0]
print(file_name)

# '''Load a json file contains bounding boxes'''
# json_all = {'version': '4.4.0', 'objects': [], 'background': '#eee'}
json_dict = []
json_all = None
if os.path.isfile('json/' + file_name + '.json'):
    fileObject = open('json/' + file_name + '.json', "r")
    jsonContent = fileObject.read()
    json_dict = json.loads(jsonContent)

    json_all = {'version': '4.4.0', 'objects': [], 'background': '#eee'}
    json_all['objects'] = json_dict

else:
    print('clear')
    if ss.clear:
        json_all = {'version': '4.4.0', 'objects': [{'type': 'rect', 'version': '4.4.0', 'originX': 'left', 'originY': 'top', 'left': 1, 'top': 1, 'width': 1, 'height': 1, 'fill': 'rgba(255, 165, 0, 0.3)', 'stroke': '#000000', 'strokeWidth': 3, 'strokeDashArray': None, 'strokeLineCap': 'butt', 'strokeDashOffset': 0, 'strokeLineJoin': 'miter', 'strokeUniform': True, 'strokeMiterLimit': 4, 'scaleX': 1, 'scaleY': 1, 'angle': 0, 'flipX': False, 'flipY': False, 'opacity': 1, 'shadow': None, 'visible': True, 'backgroundColor': '', 'fillRule': 'nonzero', 'paintFirst': 'fill', 'globalCompositeOperation': 'source-over', 'skewX': 0, 'skewY': 0, 'rx': 0, 'ry': 0}], 'background': '#eee'}
        ss.clear=False
        st.experimental_rerun()
    else:
        json_all=None

        # json_all['objects'] = None

print(json_all)

canvas_result = st_canvas(
    fill_color="rgba(255, 100, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(ss.file_list[ss.counter]),
    update_streamlit=True,
    height=451,
    drawing_mode=btn_dr_mode,
    initial_drawing=json_all,
    display_toolbar=True,
    key=0
)


def write_json():
    json_dict = canvas_result.json_data["objects"]
    jsonString = json.dumps(json_dict)
    jsonFile = open('json/' + file_name + '.json', "w")
    jsonFile.write(jsonString)


col1, col2 = st.beta_columns(2)
btn_next = col1.button('Next')
if btn_next:
    write_json()
    ss.counter += 1
    canvas_result.json_data["objects"] = None
    ss.clear = True
    st.experimental_rerun()


btn_prev = col2.button('Prev')
if btn_prev:
    write_json()
    ss.counter -= 1
    canvas_result.json_data["objects"] = None
    ss.clear = True
    st.experimental_rerun()

