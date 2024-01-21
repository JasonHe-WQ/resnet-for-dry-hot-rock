import model_training
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import dataset


def eval_model(dataset,model_path):
    # 加载数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model, criterion, _, _, _ = model_training.init_model(model_path,dataloader)
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
            inputs_cuda = inputs_cuda.unsqueeze(1).to(torch.float32)
            outputs = model(inputs_cuda)
            loss = criterion(outputs, labels_cuda)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_val_loss = val_loss / len(dataloader)
    print(f"Val Loss: {avg_val_loss}")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


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
    eval_model(dataset=dataset,model_path='./best_model.pth')
