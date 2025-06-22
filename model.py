# model.py
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

class DigitGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(DigitGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        x = torch.cat([noise, labels], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

def train_generator():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=128, shuffle=True)

    model = DigitGenerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(2): 
        for images, labels in loader:
            batch_size = images.size(0)
            noise = torch.randn(batch_size, 100)
            fake_images = model(noise, labels)
            loss = criterion(fake_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'digit_gen.pth')
    return model

train_generator()