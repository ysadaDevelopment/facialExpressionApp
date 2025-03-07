import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision.datasets import ImageFolder
from model.facialemotion import EmotionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_expression(model,frame, class_labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = transforms.ToTensor()(face).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(face)
            _, predicted = torch.max(output, 1)
            label = class_labels[predicted.item()]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame


def main():
    #Get Labels 
    # Load dataset (Use FER-2013 or a custom dataset folder)
    dataset_path = "./dataset"  # Replace with actual dataset path
    train_dataset = ImageFolder(root=dataset_path + "/train")
    class_labels = train_dataset.classes  # Get emotion labels

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("facialemotionmodel.pth"))

    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = predict_expression(model,frame,class_labels)
        cv2.imshow("Facial Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


    

