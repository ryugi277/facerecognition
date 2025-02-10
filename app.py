import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image

# Judul aplikasi
st.set_page_config(page_title="Gender Prediction System", page_icon="👦👧")
st.title("👦👧 Gender Prediction System")

# Sidebar untuk memilih input
st.sidebar.header("📥 Input Data")
input_type = st.sidebar.radio("Choose Input Method:", ["Upload Image", "Webcam"])

# Fungsi untuk memuat model
from torchvision.models import googlenet

@st.cache_resource
def load_model():
    # Menggunakan GoogLeNet tanpa auxiliary logits
    model = googlenet(pretrained=False, aux_logits=False)  # Nonaktifkan auxiliary logits
    model.fc = nn.Linear(model.fc.in_features, 2)  # Output untuk 2 kelas: Male dan Female
    model.load_state_dict(torch.load("googleNet.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


# Memuat model
model = load_model()

# Fungsi untuk preprocess gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Fungsi untuk prediksi gender
def predict_gender(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        prediction = model(input_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()
    classes = ["Male", "Female"]
    return classes[predicted_class]

# Jika input berupa Upload Image
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            # Membuka dan menampilkan gambar
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Prediksi gender
            predicted_gender = predict_gender(image)
            st.write(f"### Predicted Gender: **{predicted_gender}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Jika input berupa Webcam
elif input_type == "Webcam":
    st.markdown("### Activate Webcam")
    FRAME_WINDOW = st.image([])

    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Unable to access webcam. Please ensure it's connected.")
    else:
        run = st.button("Start Webcam")
        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Unable to read from webcam.")
                break

            # Konversi frame ke RGB dan prediksi gender
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Prediksi gender
            try:
                predicted_gender = predict_gender(image)
                label = f"Gender: {predicted_gender}"
                # Tambahkan label ke frame
                cv2.putText(frame_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                break

            # Tampilkan frame
            FRAME_WINDOW.image(frame_rgb)

        cap.release()