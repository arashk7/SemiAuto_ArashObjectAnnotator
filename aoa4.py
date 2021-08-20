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
import re

stroke_width = 3
stroke_color = "#000000"
bg_color = "#eee"

# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)


out_dir = 'D:/dataset/out_db'
out_type = st.selectbox('Select output data purpose:',('train','test'))
st.write('Output directory is: '+ out_dir+'/'+out_type)

btn_inputdir = st.button('Select input image directory')
ss = SS.get(file_list='', counter=0, output_dir='', clear=False,predict=False)

if btn_inputdir:
    dir_in = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    ss.file_list = glob.glob(dir_in + "/*.jpg")

col1, col2, col3 = st.beta_columns(3)
btn_dr_mode = col1.radio('Drawing mode:', ('rect', 'transform'))
btn_class = col2.radio('Selected class:', ('person','forklift'))
btn_loadBB = col3.radio('Default BB from:', ('JSON','Predict'))
# print(ss.file_list[ss.counter])
bg_image = cv2.imread(ss.file_list[ss.counter])
img_w = bg_image.shape[1]
img_h = bg_image.shape[0]
print(str(img_w)+' '+str(img_h))
basename = os.path.basename(os.path.normpath(ss.file_list[ss.counter]))

file_name = basename.split('.')[0]
# print(file_name)

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
        json_all = {'version': '4.4.0', 'objects': [{'type': 'rect', 'version': '4.4.0', 'originX': 'left', 'originY': 'top', 'left': 1, 'top': 1, 'width': 1, 'height': 1, 'fill': 'rgba(255, 165, 0, 0.3)', 'stroke': '#000000', 'strokeWidth': 3, 'strokeDashArray': None, 'strokeLineCap': 'butt', 'strokeDashOffset': 0, 'strokeLineJoin': 'miter', 'strokeUniform': True, 'strokeMiterLimit': 4, 'scaleX': 1, 'scaleY': 1, 'angle': 0, 'flipX': False, 'flipY': False, 'opacity': 1, 'shadow': None, 'visible': True, 'backgroundColor': '', 'fillRule': 'nonzero', 'paintFirst': 'fill', 'globalCompositeOperation': 'source-over', 'skewX': 0, 'skewY': 0, 'rx': 0, 'ry': 0,'class':'person'}], 'background': '#eee'}
        ss.clear=False
        st.experimental_rerun()
    else:
        json_all=None



if btn_class=='person':
    fill_color = "rgba(255, 0, 0, 0.3)"
else:
    fill_color = "rgba(0, 255, 0, 0.3)"
def color_to_class(color_str):
    out = 'test'
    str = re.findall(r'[\d\.\d]+',color_str)
    print(str)
    if int(str[0])==255 and int(str[1])==0 and int(str[2])==0:
        out = 'person'
    elif int(str[0]) == 0 and int(str[1]) == 255 and int(str[2]) == 0:
        out = 'forklift'
    elif int(str[0]) == 0 and int(str[1]) == 0 and int(str[2]) == 255:
        out = 'box'
    return out
def class_to_id(class_str):
    classes=['person','forklift']
    return classes.index(class_str)
# print(color_to_class(fill_color))

canvas_result = st_canvas(
    fill_color=fill_color,  # Fixed fill color with some opacity
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

from shutil import copyfile
def write_json():
    json_dict = canvas_result.json_data["objects"]
    jsonString = json.dumps(json_dict)
    jsonFile = open('json/' + file_name + '.json', "w")
    jsonFile.write(jsonString)

def copy_img():
    src = ss.file_list[ss.counter]
    dst = out_dir+'/'+out_type+'/images/'+basename
    copyfile(src,dst)

def write_label_txt():
    dst = out_dir +'/'+out_type+ '/labels/' + file_name+'.txt'
    file_txt = open(dst,'w')
    dict_obj=canvas_result.json_data["objects"]

    for key in dict_obj:
        print(key['fill'])
        cls_id=class_to_id(color_to_class(key['fill']))
        x=key['left']
        y=key['top']
        w=key['width']
        h=key['height']
        if x<0 or y<0:
            continue

        fw = 601
        fh = 451

        rw= w/fw
        rh = h/fh


        print(str(x)+' '+str(y))
        L=str(cls_id)+' '+str(x/fw+rw/2)+' '+str(y/fh+rh/2)+' '+str(rw)+' '+str(rh)+'\n'

        file_txt.writelines(L)
    file_txt.close()

col1, col2 = st.beta_columns(2)
btn_prev = col1.button('Save & Prev')
if btn_prev:
    write_json()
    copy_img()
    write_label_txt()
    ss.counter -= 1
    canvas_result.json_data["objects"] = None
    st.experimental_rerun()

btn_next = col2.button('Save & Next')
if btn_next:
    write_json()
    copy_img()
    write_label_txt()
    ss.counter += 1
    canvas_result.json_data["objects"] = None
    st.experimental_rerun()

btn_train = col1.button('Train')
if btn_train:
    st.write('Wait for training to be finished...')
    os.system('python ./yolov5/train.py --img 300 --batch 8 --epochs 20 --weights ../yolov5s.pt --data '+ out_dir+'/data.yaml')
    st.experimental_rerun()

btn_predict = col2.button('perdict')
if btn_predict:
    st.write('predict')
    # ss.predict = True
    st.write('Wait for predicting to be finished...')
    os.system(
        'python ./yolov5/detect.py --source '+out_dir+'..\Dataset_Zhitang_Yolo5_3/test\images --weights runs/train\exp14\weights/best.pt --conf 0.25 --save-txt')
    st.experimental_rerun()