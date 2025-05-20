import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7

model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model.load_state_dict(torch.load("resnet18_facial_expression.pth"))
model.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model başarıyla yüklendi ve kullanıma hazır.")

