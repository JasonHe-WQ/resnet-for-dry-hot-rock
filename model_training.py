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
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 2)  # 二分类

        self.resnet50 = resnet50

    def forward(self, x):
        return self.resnet50(x)


def init_model(model_load_path,val_loader):
    model = ModifiedResNet50().cuda()
    # model.compile()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    try:
        model.load_state_dict(torch.load(model_load_path))
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
        best_val_loss = float('inf')
        print(e)
        print("Training without loading weights")
    print("initialization finished")
    return model, criterion, optimizer, scheduler, best_val_loss


def train_and_validate_model(model, train_loader, val_loader,
                             criterion, optimizer, scheduler,
                             best_model_save_path,best_val_loss,num_epochs=8000,):
    for epoch in range(num_epochs):
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_save_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")


if __name__ == '__main__':
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

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model, criterion, optimizer, scheduler, best_val_loss = init_model('./best_model.pth',val_loader)
    train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                             best_model_save_path='./best_model.pth', best_val_loss=best_val_loss, num_epochs=8000)
