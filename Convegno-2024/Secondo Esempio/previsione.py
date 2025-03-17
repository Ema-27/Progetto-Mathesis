import random
import time  # Importa il modulo time per aggiungere un ritardo tra le richieste
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# Caricamento del modello
class DogCatClassifier(nn.Module):
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = DogCatClassifier()
model.load_state_dict(torch.load('dog_cat_classifier.pth'))
model.eval()

# Preprocessing delle immagini
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])


# Funzione per la previsione dell'immagine
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = 'dog' if output.item() > 0 else 'cat'
    return predicted_class


# Funzione per visualizzare l'immagine con la previsione
def visualize_prediction(image_path):
    predicted_class = predict_image(image_path)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(predicted_class)
    plt.show()


# Funzione per conteggiare le immagini corrette
def count_correct_predictions(image_path, real_category):
    predicted_class = predict_image(image_path)
    return predicted_class == real_category


# Inizializzazione del contatore per le immagini corrette
correct_predictions_dog = 0
correct_predictions_cat = 0

# Esempio di utilizzo
for i in range(20):
    real_category = random.choice(['dog', 'cat'])
    image_number = random.randint(0, 12499)
    image_path = f'Pet/{real_category.capitalize()}/{image_number}.jpg'
    visualize_prediction(image_path)
    if real_category == 'dog':
        correct_predictions_dog += count_correct_predictions(image_path, real_category)
    else:
        correct_predictions_cat += count_correct_predictions(image_path, real_category)

    # Aggiungi un ritardo tra le richieste per evitare errori 429 Too Many Requests
    time.sleep(1)  # Ritardo di 1 secondo tra le richieste

print("Numero totale di immagini corrette per i cani:", correct_predictions_dog,"/11")
print("Numero totale di immagini corrette per i gatti:", correct_predictions_cat,"/9")
