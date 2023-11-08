import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import data_preprocessing as dp
import data
from Models import SimpleNN

num_epochs = 1000

def calculate_accuracy(outputs, targets):
    # 获取每个时间点概率最高的类别
    _, predicted = torch.max(outputs, 2)  # 返回最大值所在的索引，即预测的类别
    # 计算正确预测的数量
    correct = (predicted == targets).sum().item()
    # 计算精度
    accuracy = 100 * correct / targets.numel()  # numel 返回张量中元素的总数
    return accuracy

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在这个代码块中不计算梯度
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)*outputs.shape[2]
            correct += (predicted == targets).sum().item()
    model.train()  # 将模型设置回训练模式
    return 100 * correct / total


if __name__ == '__main__':
    inputs, targets = dp.get_data()
    train_data_input = torch.from_numpy(inputs[0:16]).transpose(1, 2)
    train_data_target = torch.from_numpy(targets[0:16])
    test_data_input = torch.from_numpy(inputs[16:]).transpose(1, 2)
    test_data_target = torch.from_numpy(targets[16:])
    dataset_train = data.MyDataset(train_data_input.float(), train_data_target.float())
    dataset_test = data.MyDataset(test_data_input.float(),test_data_target.float())

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=1,shuffle=True)

    num_channels = 12
    # 类别的数量
    num_classes = 9
    # 创建模型
    model = SimpleNN.SimpleCNN(num_channels, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    with open('./logs/training_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_accuracy,test_accuracy\n')

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算训练集和测试集的精度
        train_accuracy = calculate_accuracy(dataloader_train, model)
        test_accuracy = calculate_accuracy(dataloader_test, model)

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {running_loss / len(dataloader_train)}, '
              f'Train Accuracy: {train_accuracy}%, '
              f'Test Accuracy: {test_accuracy}%')

        # 将训练信息写入文件
        with open('./logs/training_log.csv', 'a') as f:
            f.write(f'{epoch + 1},{running_loss / len(dataloader_train)},'
                    f'{train_accuracy},{test_accuracy}\n')

    print('Finished Training')
