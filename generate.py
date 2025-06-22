# generate.py
import torch
from model import DigitGenerator

def load_generator():
    model = DigitGenerator()
    model.load_state_dict(torch.load("digit_gen.pth"))
    model.eval()
    return model

def generate_digit_images(digit: int, count: int = 5):
    model = load_generator()
    noise = torch.randn(count, 100)
    labels = torch.tensor([digit] * count)
    with torch.no_grad():
        images = model(noise, labels)
    return images  # shape: [count, 1, 28, 28]

if __name__ == "__main__":
    digit = 5  # Example digit to generate
    count = 5  # Number of images to generate
    images = generate_digit_images(digit, count)
    print(f"Generated {count} images for digit {digit}.")
    print(images.shape)  # Should print: torch.Size([count, 1, 28, 28])