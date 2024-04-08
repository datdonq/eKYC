import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Visualize ảnh và vẽ thông tin lên")
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc ảnh từ file tải lên
        image = read_image(uploaded_file)

        # Hiển thị ảnh gốc
        st.subheader("Ảnh gốc:")
        st.image(image, caption="Ảnh gốc", use_column_width=True)

        # Vẽ thông tin lên ảnh
        annotated_image = annotate_image(image)

        # Hiển thị ảnh đã được vẽ thông tin
        st.subheader("Ảnh đã vẽ thông tin:")
        st.image(annotated_image, caption="Ảnh đã vẽ thông tin", use_column_width=True)

def read_image(uploaded_file):
    # Đọc ảnh từ file tải lên bằng OpenCV
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    return image

def annotate_image(image):
    # Viết code để vẽ thông tin lên ảnh tại đây
    # Ví dụ: vẽ hình chữ nhật đỏ trên ảnh
    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (50, 50), (200, 200), (0, 0, 255), 2)
    return annotated_image

if __name__ == "__main__":
    main()
