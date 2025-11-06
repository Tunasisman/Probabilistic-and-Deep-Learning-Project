import torch
import torch.nn as nn
import numpy as np

class WaistPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def predict(measurements):
    device = torch.device("cpu")

    # Load model and scalers
    bundle = torch.load("waist_model.pkl", map_location=device, weights_only=False)

    model = WaistPredictor().to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    X_mean = bundle["X_mean"]
    X_scale = bundle["X_scale"]
    y_mean = bundle["y_mean"]
    y_scale = bundle["y_scale"]

    # Normalize input
    X_np = measurements.numpy()
    X_scaled = (X_np - X_mean) / X_scale
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Predict and inverse scale
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
    y_pred_mm = y_pred_scaled * y_scale[0] + y_mean[0]

    return torch.tensor(y_pred_mm, dtype=torch.float32)
