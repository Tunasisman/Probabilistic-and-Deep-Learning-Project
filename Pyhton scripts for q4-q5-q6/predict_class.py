import torch
import torch.nn as nn
import torchvision.transforms as T

# Define the model architecture (same as training)
class ResNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        # A helper function to build a basic conv block
        def BasicBlock(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2)
            )
        # 4 convolutional blocks, each halving the spatial resolution
        self.b1 = BasicBlock(3,  32)
        self.b2 = BasicBlock(32, 64)
        self.b3 = BasicBlock(64,128)
        self.b4 = BasicBlock(128,256)

        # Fully connected layer after flattening: 256 * 4 * 4 = 4096 features
        self.fc = nn.Linear(256*4*4, 3)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = x.view(x.size(0), -1) # Flatten for FC
        return self.fc(x)         # Output logits (B, 3)

# Preprocessing — must match the validation transform used during training
preprocess = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.Normalize([0.5]*3, [0.5]*3)  # Normalization used during training
])

# Prediction function as required
def predict(images):
    """
    Args:
        images: Tensor of shape (B, 3, 256, 256), values in (0, 1)

    Returns:
        Tensor of shape (B, 1) containing predicted class labels (0, 1, or 2)
    """
    device = torch.device("cuda" if images.is_cuda else "cpu")

    # Resize and normalize the batch manually
    images = torch.nn.functional.interpolate(images, size=64, mode='bilinear', align_corners=False)
    for c in range(3):
        images[:,c] = (images[:,c] - 0.5) / 0.5  # Normalize to match training

    # Load model and weights
    model = ResNetSmall().to(device)
    model.load_state_dict(torch.load('best_garment_resnet.pkl', map_location=device))
    model.eval()

    # Run inference
    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(1, keepdim=True)

    return preds
