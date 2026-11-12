import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 2 홍길동")
st.write(":dog:")
st.sidebar.write("## my sidebar")

def process_image(upload):
    image = Image.open(upload)
    col1.write("Input")
    col1.image(image)
    output = image.convert('L')
    col2.write("Output 1: grayscale")
    col2.image(output)
    output = ImageOps.equalize(output)
    col3.write("Output 2: histeq")
    col3.image(output)
    col4.write("Output 3: color edges")
    color = image.convert('RGB')
    color.thumbnail((800, 800))                                    
    gray = color.convert('L').filter(ImageFilter.GaussianBlur(2))
    gy, gx = np.gradient(np.array(gray, dtype=float))              
    mag = np.sqrt(gx**2 + gy**2)                                   
    mask = mag > np.percentile(mag, 60)                            
    rgb = np.array(color)
    bw = np.array(color.convert('L'))
    output = np.where(mask[..., None], rgb, bw[..., None])
    col4.image(output)

col1, col2, col3, col4 = st.columns(4)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg"])

if my_upload is not None:
    process_image(upload=my_upload)
else:
    process_image("./gr.jpeg")