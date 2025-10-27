import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
import mediapipe as mp

# -----------------------------
# Hand landmark extraction
# -----------------------------
mp_hands = mp.solutions.hands

def extract_hand_landmarks_video(video_path, n_frames=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return np.zeros((n_frames, 42), dtype=np.float32)
    frame_indices = np.linspace(0, max(total_frames-1, 1), n_frames, dtype=int)
    hands_data = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                hands_data.append([0]*42)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                frame_landmarks = []
                for lm in landmarks.landmark:
                    frame_landmarks.append(lm.x)
                    frame_landmarks.append(lm.y)
                hands_data.append(frame_landmarks)
            else:
                hands_data.append([0]*42)
    
    cap.release()
    while len(hands_data) < n_frames:
        hands_data.append([0]*42)
    return np.array(hands_data[:n_frames], dtype=np.float32)

# -----------------------------
# Load trained model
# -----------------------------
class SignLSTM(torch.nn.Module):
    def __init__(self, input_size=42, hidden_size=64, num_layers=2, num_classes=20):
        super(SignLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignLSTM(num_classes=20).to(device)
model.load_state_dict(torch.load("asl_lstm_model.pth", map_location=device))
model.eval()

# -----------------------------
# Label mapping
# -----------------------------
labels = ['book','drink','computer','chair','clothes','candy','cousin','year','go','walk',
          'help','deaf','fine','thin','black','who','before','no','yes','all']

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("Sign Language Detection App")

option = st.sidebar.selectbox("Choose Input Type", ["Upload Video"])
st.sidebar.write("Upload a video to predict the sign.")

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if uploaded_file:
        # Save temporarily
        tfile = Path("temp_video.mp4")
        with open(tfile, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(str(tfile))

        if st.button("Predict Sign"):
            st.info("Processing video, please wait...")
            X = extract_hand_landmarks_video(str(tfile))
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(X_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                st.success(f"Predicted Sign: **{labels[pred_idx]}**")
