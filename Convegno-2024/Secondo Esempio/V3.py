import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

# Definizione del modello
class CustomDogCatBirdClassifier(nn.Module):
    def __init__(self, num_classes=2):  # Modifica qui il numero di classi
        super(CustomDogCatBirdClassifier, self).__init__()
        # Caricamento di una rete pre-addestrata (ResNet18) e modifica dell'ultimo strato completamente connesso
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # L'ultimo strato ora ha il numero corretto di classi

    def forward(self, x):
        x = self.resnet(x)
        return x

# Funzione per il training del modello
def train_model(model, criterion, optimizer, dataloaders, num_epochs=20):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc * 100))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Preprocessing delle immagini
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Caricamento del dataset
data_dir = "PetImages"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=25, shuffle=True, num_workers=0) for x in ['train', 'test']}

# Verifica disponibilità GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definizione del modello, della funzione di perdita e dell'ottimizzatore
model = CustomDogCatBirdClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Avvio del training del modello
train_model(model, criterion, optimizer, dataloaders, num_epochs=20)

# Salva il modello addestrato
torch.save(model.state_dict(), 'custom_model.pth')

# Funzione per predire e visualizzare singole immagini
def visualize_prediction(model, class_names, img_path):
    model.eval()
    image = Image.open(img_path)
    image_tensor = data_transforms['test'](image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    prediction = class_names[predicted[0]]
    plt.imshow(image)
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.show()

# Dizionario per mappare gli indici di classe predetti ai nomi delle classi
class_names = image_datasets['train'].classes

# Esempio di predizione e visualizzazione di un'immagine
example_image_path = "Pet/Dog/0.jpg"  # Percorso dell'immagine di esempio
visualize_prediction(model, class_names, example_image_path)
