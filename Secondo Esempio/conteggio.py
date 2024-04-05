import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Disabilita i warning
import warnings
warnings.filterwarnings("ignore")

# Caricamento del modello addestrato
class CustomDogCatBirdClassifier(nn.Module):
    def __init__(self):
        super(CustomDogCatBirdClassifier, self).__init__()
        # Caricamento di una rete pre-addestrata (ResNet18) e modifica dell'ultimo strato completamente connesso
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # L'ultimo strato ora ha 2 classi: cane, gatto

    def forward(self, x):
        x = self.resnet(x)
        return x

# Caricamento del modello e impostazione in modalità valutazione
model = CustomDogCatBirdClassifier()
model.load_state_dict(torch.load('custom_model.pth'))
model.eval()

# Preprocessing delle immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Converti l'immagine in scala di grigi con 3 canali
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])  # Utilizza media e deviazione standard per un singolo canale
])

# Funzione per la previsione dell'immagine
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Restituisce l'etichetta predetta

# Inizializzazione delle liste per il conteggio delle immagini
total_images = [0, 0]
correct_predictions = [0, 0]

# Esempio di utilizzo
for i in range(5000):
    category = random.choice(['Dog', 'Cat'])
    total_images[0 if category == 'Dog' else 1] += 1

    image_number = random.randint(0, 12499)
    image_path = f'Pet/{category.capitalize()}/{image_number}.jpg'
    predicted_label = predict_image(image_path)
    if predicted_label == 1:
        predicted_class = 'Dog'
    else:
        predicted_class = 'Cat'

    if predicted_class == category:
        correct_predictions[0 if category == 'Dog' else 1] += 1

print("Numero totale di immagini per classe:", total_images)
print("Predizioni corrette per classe:", correct_predictions)
