# ========================
# Imports et configuration initiale
# ========================
import os
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tqdm import tqdm
import splitfolders
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# Paramètres globaux
# ========================
ROOT_DIR = './eeg_raw_data'           
OUTPUT_DIR = './dataset_images'           
DATA_DIR = './dataset_split'
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Mappings des labels par session (tiré du README SEED-IV)
# ========================
label_map = {0: 'neutral', 1: 'sadness', 2: 'fear', 3: 'joy'}
session_labels = {
    '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

# ========================
# Génération de spectrogrammes depuis les signaux EEG (.mat)
# ========================
def save_spectrogram_image(signal, save_path):
    f, t, Sxx = spectrogram(signal, fs=200)
    Sxx = np.log(Sxx + 1e-10)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_images_all_sessions():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for session in ['1', '2', '3']:
        session_path = os.path.join(ROOT_DIR, session)
        for file in tqdm(os.listdir(session_path), desc=f"Session {session}"):
            if file.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(session_path, file))
                for i in range(1, 25):
                    key = f'tyc_eeg{i}'
                    if key not in mat:
                        continue
                    eeg_trial = mat[key]
                    label_id = session_labels[session][i - 1]
                    label = label_map[label_id]
                    for ch_idx, signal in enumerate(eeg_trial):
                        label_dir = os.path.join(OUTPUT_DIR, label)
                        os.makedirs(label_dir, exist_ok=True)
                        img_name = f"{file[:-4]}_trial{i}_ch{ch_idx}.png"
                        save_path = os.path.join(label_dir, img_name)
                        save_spectrogram_image(signal, save_path)

# ========================
# Split automatique des données (train/val/test)
# ========================
def split_dataset():
    splitfolders.ratio(OUTPUT_DIR, output=DATA_DIR, seed=1337, ratio=(.7, .2, .1))

# ========================
# Transformations d'image pour ResNet
# ========================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

# ========================
# Chargement des données avec PyTorch
# ========================
def load_data():
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val', 'test']}
    return image_datasets, dataloaders

# ========================
# Modèle ResNet18 (transfer learning)
# ========================
def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# ========================
# Entraînement du modèle + tracé des courbes
# ========================
def train_model(model, dataloaders, image_datasets):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\n📘 Époque {epoch + 1}/{NUM_EPOCHS}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"🔹 {phase} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_model.pth')
    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# ========================
# Tracé des courbes d'apprentissage
# ========================
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========================
# Évaluation sur les données de test
# ========================
def evaluate_model(model, dataloaders, class_names):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n📊 Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()

# ========================
# Point d'entrée principal du script
# ========================
if __name__ == "__main__":
    generate_images_all_sessions()
    print("\n✅ Tous les spectrogrammes ont été générés.")
    split_dataset()
    image_datasets, dataloaders = load_data()
    class_names = image_datasets['train'].classes
    model = build_model()
    train_model(model, dataloaders, image_datasets)
    evaluate_model(model, dataloaders, class_names)
