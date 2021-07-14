import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "freedraw", "line", "circle", "transform", "polygon")
)

st.button('next')
st.button('prev')
realtime_update = st.sidebar.checkbox("Update in realtime", True)
# json_dict = {'id':1,'type': 'rect', 'version': '4.4.0', 'originX': 'left', 'originY': 'top', 'left': 81, 'top': 113, 'width': 89,
#      'height': 79, 'fill': 'rgba(255, 165, 0, 0.3)', 'stroke': '#000000', 'strokeWidth': 3, 'strokeDashArray': None,
#      'strokeLineCap': 'butt', 'strokeDashOffset': 0, 'strokeLineJoin': 'miter', 'strokeUniform': True,
#      'strokeMiterLimit': 4, 'scaleX': 1, 'scaleY': 1, 'angle': 0, 'flipX': False, 'flipY': False, 'opacity': 1,
#      'shadow': None, 'visible': True, 'backgroundColor': '', 'fillRule': 'nonzero', 'paintFirst': 'fill',
#      'globalCompositeOperation': 'source-over', 'skewX': 0, 'skewY': 0, 'rx': 0, 'ry': 0}
json_all = {'version': '4.4.0', 'objects': [], 'background': '#eee'}
json_dict = []
fileObject = open("data.json", "r")
jsonContent = fileObject.read()
json_dict = json.loads(jsonContent)

# json_all['objects'].append(objects_dict2)
json_all['objects'] = json_dict
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=True,
    height=451,
    drawing_mode='drawing_mode',
    initial_drawing=json_all,
    display_toolbar=True,
    key="canvas0",
)
# Create a canvas component
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=451,
#     drawing_mode=drawing_mode,
#     display_toolbar=st.sidebar.checkbox("Display toolbar", True),
#     initial_drawing=canvas_result_0.json_data,
#     key="canvas1",
# )
# canvas_result.json_data['objects']=[]
# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

if(st.button('save')):
    json_dict = canvas_result.json_data["objects"]
    jsonString = json.dumps(json_dict)
    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)

if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
if canvas_result.json_data is not None:
    data = pd.json_normalize(canvas_result.json_data["objects"])
    json_dict=canvas_result.json_data["objects"]
    jsonString = json.dumps(json_dict)
    # jsonFile = open("data.json", "w")
    # jsonFile.write(jsonString)
    # jsonFile.close()

    # print(canvas_result.json_data)
    # for d in data:
    X = data['left']
    Y = data['top']
    st.write(X[0])
    st.write(Y[0])
