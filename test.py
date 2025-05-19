import torch
import torch.nn as nn
from torchvision import models

# Sınıf sayısını tanımla
NUM_CLASSES = 7

# Model mimarisini oluştur
model = models.resnet18(weights=None)  # pretrained değil
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Ağırlıkları yükle
model.load_state_dict(torch.load("resnet18_facial_expression.pth"))
model.eval()  # Modeli değerlendirme moduna al

# Cihaz ayarı (GPU varsa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model başarıyla yüklendi ve kullanıma hazır.")

