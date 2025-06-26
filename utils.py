import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Modelo simples
class RouteOptimizerNet(nn.Module):
    def __init__(self):
        super(RouteOptimizerNet, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)  # Saída: delta_x, delta_y da próxima posição

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Gerador de dados sintéticos para simulação
def get_data_loaders():
    def generate_sample():
        temperature = random.uniform(15, 40)
        altitude = random.uniform(50, 300)
        wind_speed = random.uniform(0, 20)
        wind_dir = random.uniform(0, 360)
        area_covered = random.uniform(0, 100)
        pos_x = random.uniform(0, 500)
        pos_y = random.uniform(0, 500)

        # Saída: próximo delta x e delta y

        delta_x = random.uniform(-10, 10)
        delta_y = random.uniform(-10, 10)
        features = [temperature, altitude, wind_speed, wind_dir, area_covered, pos_x, pos_y]
        labels = [delta_x, delta_y]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    dataset = [generate_sample() for _ in range(1000)]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    return train_loader, test_loader
