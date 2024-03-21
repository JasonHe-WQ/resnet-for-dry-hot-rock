# diyiwei batch size demension(h,w)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_files, label_files):
        self.data = []
        self.labels = []

        for data_file, label_file in zip(data_files, label_files):
            df = pd.read_csv(data_file, header=None)
            labels_df = pd.read_csv(label_file, header=None)
            print(df.shape, "data shape of file", data_file)
            print(labels_df.shape, "label shape of file", label_file)

            for i in range(len(df)):
                for start in range(50, 1550, 200):
                    data_segment = df.iloc[i, start:start + 100].values
                    # 每行8个波形，每个波形100个点
                    # 将1x100的数据重新塑形为10x10
                    reshaped_segment = data_segment.reshape(10, 10)
                    self.data.append(reshaped_segment)
                    self.labels.append(labels_df.iloc[i, start // 100])

        print(self.data[0].shape, "data shape")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    # 加载数据
    data_files = ['./-10dB/-10dB.csv']
    label_files = ['./-10dB/-10dB-labels.csv']
    dataset = CustomDataset(data_files, label_files)

    # 划分训练集和验证集
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
