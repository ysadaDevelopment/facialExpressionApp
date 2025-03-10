import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from model.facialemotion import EmotionCNN
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

allowed_origins = [
    "https://jazzy-frangipane-2448a0.netlify.app",  # If hosted on Netlify
    "https://your-frontend.vercel.app",   # If hosted on Vercel
    "https://your-username.github.io",    # If using GitHub Pages
    "http://localhost:5500",              # If testing locally with Live Server
    "http://127.0.0.1:5500"
]
# Add CORS Middleware (Allow All Origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows all origins (set specific origins for security)
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("facialemotionmodel.pth", map_location=device))
model.eval()

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion Labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Image Transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.get("/")
def home():
    return {"message": "Facial Expression Detection API is running!"}


@app.post("/predict/")
async def predict_expression(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return {"error": "No face detected"}

    x, y, w, h = faces[0]  # Use first detected face
    face = gray[y:y+h, x:x+w]

    # Convert to PIL and transform
    face_pil = Image.fromarray(face)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    # Predict expression
    with torch.no_grad():
        output = model(face_tensor)
        _, predicted = torch.max(output, 1)
        label = class_labels[predicted.item()]

    return {"emotion": label}


@app.get("/video_feed")
def video_feed():
    """
    Opens a real-time webcam video feed for facial expression detection.
    Use this in a separate script or endpoint.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = transforms.ToTensor()(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face)
                _, predicted = torch.max(output, 1)
                label = class_labels[predicted.item()]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Facial Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
