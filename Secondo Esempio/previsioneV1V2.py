import random
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from urllib.error import HTTPError

# Disabilita i warning
import warnings

warnings.filterwarnings("ignore")


# Caricamento del modello addestrato
class CustomDogCatClassifier(nn.Module):
    def __init__(self):
        super(CustomDogCatClassifier, self).__init__()
        # Caricamento di una rete pre-addestrata (ResNet18) e modifica dell'ultimo strato completamente connesso
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # L'ultimo strato ora ha 2 classi: cane, gatto

    def forward(self, x):
        x = self.resnet(x)
        return x


# Caricamento del modello e impostazione in modalità valutazione
model = CustomDogCatClassifier()
model.load_state_dict(torch.load('custom_model1.pth'))
model.eval()

# Preprocessing delle immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3]),  # Escludi il canale alfa se presente
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Funzione per la previsione dell'immagine
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Restituisce l'etichetta predetta


# Funzione per visualizzare l'immagine con la previsione
def visualize_prediction(image_path):
    predicted_label = predict_image(image_path)
    if predicted_label == 1:
        predicted_class = 'Dog'
    else:
        predicted_class = 'Cat'
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(predicted_class)
    plt.show()

    return predicted_class


# Inizializzazione delle liste per il conteggio delle immagini
total_images = [0, 0]
correct_predictions = [0, 0]

# Esempio di utilizzo
for i in range(70):
    category = random.choice(['Dog', 'Cat'])
    total_images[0 if category == 'Dog' else 1] += 1

    image_number = random.randint(0, 12499)
    image_path = f'Pet/{category.capitalize()}/{image_number}.jpg'

    try:
        # Introduci un ritardo per evitare errori di "Too Many Requests"
        time.sleep(2)  # Aspetta 2 secondi tra le richieste

        predicted_class = visualize_prediction(image_path)

        if predicted_class == category:
            correct_predictions[0 if category == 'Dog' else 1] += 1

    except HTTPError as e:
        if e.code == 429:
            print("Too Many Requests. Attendere qualche istante e riprovare.")
            time.sleep(5)  # Riprova dopo 10 secondi

print("Numero totale di immagini per classe:", total_images)
print("Predizioni corrette per classe:", correct_predictions)
