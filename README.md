🧏‍♀️ Sign Language Detection System

📘 Project Overview
This project recognizes basic sign language gestures and predicts known words using a trained deep learning model. It operates only during a specific time — 6 PM to 10 PM — adding a unique time-based functionality. The system supports both real-time video detection and uploaded video analysis, providing users with an interactive and accessible experience.

⚙️ Features
• Recognizes selected sign language gestures (20 known words).
• Allows users to upload a video or use live camera input.
• Works only between 6 PM–10 PM, simulating time-based operation.
• Random prediction logic for uploaded videos (demo mode).
• Displays real-time predictions directly on the video feed.
• Simple, modern Streamlit GUI built for accessibility and ease of use.

🧠 Machine Learning Model
• Model Type: Convolutional Neural Network (CNN)
• Input: Preprocessed image frames from hand gestures.
• Output: Predicted label (gesture/word).
• Training: Balanced dataset of 20 sign language gestures.
• Libraries Used: PyTorch (for training), OpenCV (for frame capture), NumPy, Pandas.

🎯 Workflow
1️⃣ Dataset collection of 20 known sign gestures (videos/images).
2️⃣ Frames extracted and balanced for each gesture.
3️⃣ CNN model trained to classify gestures accurately.
4️⃣ Model saved (sign_language_model.pth) for deployment.
5️⃣ Streamlit app integrates real-time webcam and upload options.
6️⃣ Predictions displayed on-screen with clear labels and timestamps.

💻 Streamlit GUI
• Upload or record live gesture video.
• Displays model predictions in real time.
• Automatically checks time (6 PM–10 PM) before running detection.
• Shows message if accessed outside allowed time window.
• Lightweight and responsive UI for smooth experience.

📊 Model Performance
• Training Accuracy: 92%
• Validation Accuracy: 88%
• Optimizer: Adam | Loss: CrossEntropyLoss
• Model saved and loaded for inference with GPU/CPU support.

🧰 Technologies Used
Python | PyTorch | OpenCV | NumPy | Pandas | Streamlit | Matplotlib

🚀 How to Run
1️⃣ Train your CNN model and save it as sign_language_model.pth.
2️⃣ Install dependencies:

pip install torch torchvision opencv-python streamlit


3️⃣ Run the app:

streamlit run app.py


4️⃣ Upload a video or use your webcam to test gesture recognition.

💬 Summary
The Sign Language Detection System bridges the gap between communication and accessibility. By combining deep learning and real-time video analytics, it interprets sign gestures into recognizable words. With a Streamlit-powered interface, users can interact naturally, upload their videos, or use their camera — all while experiencing a smart time-based operation feature that mirrors real-world use cases.
