"""uavFL: A Flower / PyTorch app."""

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class DroneTrajectoryDataset(Dataset):
    def __init__(self, csv_path, map_size):
        self.data = pd.read_csv(csv_path, sep=';')
        self.map_size = map_size
        self.drones = []
        self._preprocess()
    
    def _normalize_data(self, df):
        # Position x and position y are global asbolute coordinates, get only digits that change (relevant for drones)
        df['position_x'] = df['position_x'].apply(lambda x : (x*10000) % 100)
        df['position_y'] = df['position_y'].apply(lambda x : (x*10000) % 100)
        # Normalize coordinates in relation to map size
        df['position_x'] = (df['position_x']-df['position_x'].min())/(df['position_x'].max()-df['position_x'].min())
        df['position_x'] = df['position_x'] * self.map_size
        df['position_y'] = (df['position_y']-df['position_y'].min())/(df['position_y'].max()-df['position_y'].min())
        df['position_y'] = df['position_y'] * self.map_size
        # Get battery level in percentage
        df['battery_voltage'] = df['battery_voltage'].apply(lambda x : x/52000)
        # Remove negativers from altitude
        df['altitude'] = df['altitude'].mask(df['altitude'] < 0, 0)

        return df         

    def _preprocess(self):
        # Normalize used values
        self.data = self._normalize_data(self.data)
        grouped = self.data.groupby('uid')
        for uid, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            self.drones.append(group)

    def __getitem__(self, idx):
            return self.data[idx]

    def __len__(self):
        return len(self.data)

def get_dataloader(csv_path, map_size, batch_size=32, shuffle=True):
    dataset = DroneTrajectoryDataset(csv_path, map_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
