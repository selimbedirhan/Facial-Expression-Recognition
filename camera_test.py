import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# Sınıf sayısı ve sınıf isimleri
NUM_CLASSES = 7
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Modeli yükle
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("resnet18_facial_expression.pth", map_location=torch.device('cpu')))
model.eval()

# Görüntü dönüşüm işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Kamerayı aç
cap = cv2.VideoCapture(0)
print("Kameradan görüntü alınıyor. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    # Görüntüyü kopyala ve dönüştür
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0)

    # Tahmin yap
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Tahmini görüntüye yaz
    cv2.putText(frame, f'Expression: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Facial Expression Recognition", frame)

    # Çıkış için 'q' tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
