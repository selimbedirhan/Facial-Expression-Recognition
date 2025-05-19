import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import time

# Veri dönüşümleri
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

print("Veri setleri yükleniyor...")

train_dataset = datasets.ImageFolder(root='/Users/selimbedirhanozturk/Desktop/python/dataset/train', transform=transform['train'])
test_dataset = datasets.ImageFolder(root='/Users/selimbedirhanozturk/Desktop/python/dataset/test', transform=transform['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train setinde {len(train_dataset)} örnek, Test setinde {len(test_dataset)} örnek var.")

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalıştırma cihazı: {device}")

# Model tanımlama ve transfer learning için hazır model kullanımı
print("Model hazırlanıyor...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # sınıf sayısına göre çıktı katmanı

model = model.to(device)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim fonksiyonu
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Train batch {batch_idx}/{len(loader)} - Loss: {running_loss/(batch_idx+1):.4f} - Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch tamamlandı - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

# Test fonksiyonu
def test(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Test batch {batch_idx}/{len(loader)} - Loss: {test_loss/(batch_idx+1):.4f} - Acc: {100.*correct/total:.2f}%")

    epoch_loss = test_loss / len(loader)
    epoch_acc = 100. * correct / total
    print(f"Test tamamlandı - Test Loss: {epoch_loss:.4f} - Test Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

# Eğitim döngüsü
num_epochs = 5
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} başlıyor...")
    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} tamamlandı. Süre: {elapsed:.2f} saniye.")

print("Eğitim ve test tamamlandı.")

# Modelin ağırlıklarını kaydetmek
torch.save(model.state_dict(), "resnet18_facial_expression.pth")
print("Model kaydedildi.")
