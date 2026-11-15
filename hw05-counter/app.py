import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from counting import (preprocess, make_mask, label_components,
                      colorize, overlay_numbers)

st.set_page_config(page_title="Object Counting (HW5)", layout="wide")
st.title("🔢 Object Counting")
st.caption("이미지를 업로드하면 전처리 → 임계값 → 형태학 연산 → Connected Component 순서로 객체 개수를 셉니다.")

# ---------------- Sidebar: parameters ----------------
st.sidebar.header("파라미터")
crop_top = st.sidebar.number_input("위쪽 잘라낼 픽셀 (crop)", 0, 2000, 0, step=10,
                                   help="노트북에서는 f[50:, :] 로 위 50px를 잘랐습니다.")
median_k = st.sidebar.select_slider("Median filter 크기", [3, 5, 7], value=3)
alpha = st.sidebar.slider("Edge enhancement α", 0.0, 2.0, 0.8, 0.1)

st.sidebar.subheader("Thresholding")
th_mode = st.sidebar.radio("임계값 방식", ["manual", "otsu"], horizontal=True)
th_val = st.sidebar.slider("임계값 (manual)", 0, 255, 128, disabled=(th_mode == "otsu"))
bright = st.sidebar.radio("객체 밝기", ["배경보다 밝음", "배경보다 어두움"]) == "배경보다 밝음"

st.sidebar.subheader("Morphology")
k_size = st.sidebar.select_slider("Kernel 크기", [3, 5, 7, 9], value=5)
ero = st.sidebar.slider("Erosion 반복", 0, 6, 2)
dil = st.sidebar.slider("Dilation 반복", 0, 6, 1)
min_area = st.sidebar.number_input("최소 객체 크기 (pixel, 노이즈 제거)", 0, 100000, 0,
                                   help="0이면 노트북과 동일하게 모든 component를 셉니다.")

# ---------------- Input ----------------
file = st.file_uploader("이미지 업로드 (jpg / png)", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("왼쪽 파라미터를 조절한 뒤 이미지를 업로드하세요.")
    st.stop()

gray = np.array(Image.open(file).convert("L"))
if crop_top >= gray.shape[0] - 10:
    st.error("crop 값이 이미지 높이보다 큽니다.")
    st.stop()

# ---------------- Pipeline ----------------
f, f_den, f_enh = preprocess(gray, crop_top, median_k, alpha)
th_used, mask, eroded, dilated = make_mask(f_den, th_mode, th_val, bright,
                                           k_size, ero, dil)
labels, areas, cents = label_components(dilated, min_area)
n = len(areas)

# ---------------- Result ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("검출된 객체 수", n)
c2.metric("사용된 임계값", th_used)
c3.metric("가장 큰 객체 (px)", int(areas.max()) if n else 0)
c4.metric("가장 작은 객체 (px)", int(areas.min()) if n else 0)

left, right = st.columns(2)
left.image(f, caption="입력 (grayscale, crop 적용)", use_container_width=True)
right.image(overlay_numbers(f, labels, cents),
            caption=f"결과: {n}개 (번호 = component index)", use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["전처리", "Mask / Morphology", "Components", "크기 통계"])

with tab1:
    a, b = st.columns(2)
    a.image(f_den, caption="Median filter 후", use_container_width=True)
    b.image(f_enh, caption="Edge enhancement 후", use_container_width=True)
    hist = np.bincount(f_den.ravel(), minlength=256)
    st.caption("Histogram (임계값 선택 참고)")
    st.bar_chart(pd.DataFrame({"frequency": hist}))

with tab2:
    a, b, c = st.columns(3)
    a.image(mask * 255, caption="Original mask", use_container_width=True)
    b.image(eroded * 255, caption="Erosion", use_container_width=True)
    c.image(dilated * 255, caption="Dilation", use_container_width=True)

with tab3:
    if n == 0:
        st.warning("검출된 객체가 없습니다. 임계값/객체 밝기 설정을 확인하세요.")
    else:
        st.image(colorize(labels), caption="Connected components", use_container_width=True)
        imax, imin = int(np.argmax(areas)) + 1, int(np.argmin(areas)) + 1
        a, b = st.columns(2)
        a.image((labels == imax).astype(np.uint8) * 255,
                caption=f"가장 큰 component (index {imax}, {int(areas[imax-1])} px)",
                use_container_width=True)
        b.image((labels == imin).astype(np.uint8) * 255,
                caption=f"가장 작은 component (index {imin}, {int(areas[imin-1])} px)",
                use_container_width=True)

with tab4:
    df = pd.DataFrame({"index": np.arange(1, n + 1), "pixels": areas})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"),
                       file_name="object_sizes.csv", mime="text/csv")
