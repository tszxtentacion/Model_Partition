import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,))]
)

train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 原始的前向传播函数
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def forward_1(self, x, index):
        # 提前退出前向传播函数
        # index表示中途退出的层数下标，从0（输入）开始，第index层已训练好
        i = 0
        x = x.view(x.shape[0], -1)
        if i == index:
            return x
        i += 1
        x = F.relu(self.fc1(x))
        if i == index:
            return x
        i += 1
        x = F.relu(self.fc2(x))
        if i == index:
            return x
        i += 1
        x = F.relu(self.fc3(x))
        if i == index:
            return x
        i += 1
        x = F.log_softmax(self.fc4(x), dim=1)
        if i == index:
            return x

    def forward_2(self, x, index):
        # 中间层开始进行的前向传播函数
        # 从index层开始继续推理
        i = 0
        if i >= index:
            x = x.view(x.shape[0], -1)
        i += 1
        if i >= index:
            x = F.relu(self.fc1(x))
        i += 1
        if i >= index:
            x = F.relu(self.fc2(x))
        i += 1
        if i >= index:
            x = F.relu(self.fc3(x))
        i += 1
        if i >= index:
            x = F.log_softmax(self.fc4(x), dim=1)
        return x


model = Classifier()
images, labels = next(iter(test_loader))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 1
steps = 0
train_losses, test_losses = [], []
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0.
    # validation
    with torch.no_grad():
        model.eval()    # 设置为评估模式，如关闭dropout等
        for images, labels in test_loader:
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # equals = numpy.array(equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    train_losses.append(running_loss / len(train_loader))
    test_losses.append(running_loss / len(test_loader))

    print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

# 进入推理模式
model.eval()
data_iter = iter(test_loader)
images, labels = data_iter.next()
img = images[0]
img = img.view(1, 784)  # 转换成1dim

test_loss = 0
accuracy = 0
with torch.no_grad():
    model.eval()  # 设置为评估模式，如关闭dropout等
    for images, labels in test_loader:
        # log_ps = model(images)
        # 下面两行用来替换上面一行，证明中途退出-开始的可行性
        output_mediate = model.forward_1(images, 3)
        log_ps = model.forward_2(output_mediate, 4)
        test_loss += criterion(log_ps, labels)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        # equals = numpy.array(equals)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

# # 保存模型参数
# torch.save(model.state_dict(), 'Classifier.pt')
# # 读取
# model_state_dict = torch.load('Classifier.pt')
# new_model = Classifier()
# new_model.load_state_dict(model_state_dict)
# # print(new_model)
# # 保存整个模型
# torch.save(model, 'Classifier.pt')
# new_model_ = torch.load('Classifier.pt')

