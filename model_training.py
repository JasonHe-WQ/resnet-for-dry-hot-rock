# 修改ResNet50模型
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        resnet50 = models.resnet50(weights=None)
        # 修改第一层卷积核大小和步长
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后的全连接层
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 2)  # 二分类

        self.resnet50 = resnet50

    def forward(self, x):
        return self.resnet50(x)


model = ModifiedResNet50().cuda()
# compile
model.compile()


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)  # L2正则化

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

# 最佳模型保存路径
best_model_path = './best_model.pth'

best_val_loss = float('inf')

def init():
    global best_val_loss
    try:
        # load weights
        model.load_state_dict(torch.load(best_model_path))
        # calculate the best val loss
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.unsqueeze(1).to(torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = avg_val_loss
    except Exception as e:
        # 记录最佳验证损失
        best_val_loss = float('inf')
        print(e)


# 训练和验证循环
def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=8000):
    global best_val_loss
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1).to(torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.unsqueeze(1).to(torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")


if __name__ == '__main__':
    # 加载数据
    data_files = ["./-15dB/-15dB.csv",
                  './-10dB/-10dB.csv',
                  './-5dB/-5dB.csv',
                  './0dB/0dB.csv',
                  './-5dB_modify/-5dB_modify.csv',
                  './-10dB_modify/-10dB_modify.csv',
                  './-15dB_modify/-15dB_modify.csv',
                  './0dB_modify/0dB_modify.csv']

    label_files = ['./-15dB/-15dB-labels.csv',
                   './-10dB/-10dB-labels.csv',
                   './-5dB/-5dB-labels.csv',
                   './0dB/0dB-labels.csv',
                   './-5dB_modify/-5dB_modify-labels.csv',
                   './-10dB_modify/-10dB_modify-labels.csv',
                   './-15dB_modify/-15dB_modify-labels.csv',
                   './0dB_modify/0dB_modify-labels.csv']
    dataset = dataset.CustomDataset(data_files, label_files)

    # 划分训练集和验证集
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    init()
    # 训练和验证
    train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=8000)

