# 训练数据
import time
import cv2 as cv
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import *

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 转换图片numpy array格式为tensor类型，且归一化数据
tensor_trans = transforms.ToTensor()


# 准备数据集
class GestureData(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        if train:
            self.train = True
        else:
            self.train = False

        if self.train:
            f = open(root + '/train.txt', 'r')
            print('READ TRAIN LIST FILE SUCCESS.')
        else:
            f = open(root + '/test.txt', 'r')
            print('READ TEST LIST FILE SUCCESS.')

        self.records = f.readlines()

    def __getitem__(self, index):
        # 从txt文档中获取当前index的img(tensor类型)和label(int类型)
        record = self.records[index]
        img_path = record.split(',')[0]
        label = int(record.split(',')[1].strip())
        # opencv 读取图片得到 numpy.ndarray (H x W x C)格式数据 channel为BGR
        image = cv.imread(img_path)
        # 改变图片尺寸以使得符合网络输入
        if not image.shape == (100, 100, 3):
            image = cv.resize(image, (100, 100))
        # 转换图片numpy array格式为tensor类型
        image = tensor_trans(image)

        return image, label

    def __len__(self):
        return len(self.records)


# 例化数据集
train_data = GestureData('.', train=True)
test_data = GestureData('.', train=False)
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集长度为: {}'.format(train_data_size))
print('测试数据集长度为: {}'.format(test_data_size))

# 利用 DataLoader 来加载数据集
# Batch_Size 太小，算法在 200 epoches 内不收敛。
# 随着 Batch_Size 增大，处理相同数据量的速度越快。
# 随着 Batch_Size 增大，达到相同精度所需要的 epoch 数量越来越多。
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)


# 创建网络模型
# net = MyNet()
# net = torchvision.models.alexnet(weights=None, num_classes=10)
# net = torchvision.models.vgg16(weights=None, num_classes=10)
net = torchvision.models.resnet50(weights=None, num_classes=10)
net = net.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
leaning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=leaning_rate, momentum=0.9)

# 设置训练网络的一些参数
# 记录训练的次数
total_tran_step = 0
# 记录测试的次数
total_test_step = 0

# 训练好的模型存储地址
MODEL_SAVE_PATH = 'pths'
# 添加tensorboard
writer = SummaryWriter('logs')
# 开始时间
start_time = time.time()

# 开始训练
for epoch in range(50):
    print('-----------第 {} 轮训练开始----------'.format(epoch + 1))

    # 训练步骤开始
    net.train() # 对某些网络层有效 eg: Dropout,BatchNorm
    train_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_tran_step += 1
        if i % 50 == 49:
            end_time = time.time()
            train_loss = train_loss / 50
            print('train: [epoch: %d, %5d] [loss: %.3f] [time: %.5f]' % (epoch+1, i + 1, train_loss, end_time-start_time))
            writer.add_scalar('train_loss', train_loss, total_tran_step)
            train_loss = 0.0

    # 验证步骤开始
    net.eval()  # 对某些网络有效 eg: Dropout,BatchNorm
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():  # 测试时，不需要网络模型中的梯度，不需要对梯度调整，更不需要梯度来优化
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_accuracy += (outputs.argmax(1) == labels).sum()

            # _, predicted = torch.max(outputs.data, 1)
            # test_accuracy += (predicted == labels).sum().item()

    end_time = time.time()
    print('test: [epoch: %d] [loss: %.3f, acc: %.3f] [time: %.5f]' % (epoch+1, test_loss, test_accuracy/test_data_size, end_time-start_time))
    writer.add_scalar('test_loss', test_loss, epoch+1)
    writer.add_scalar('test_accuracy', test_accuracy/test_data_size, epoch+1)

    # 保存训练模型
    torch.save(net.state_dict(), MODEL_SAVE_PATH + '/gesture_%02d.pth' % (epoch + 1))
    print('model save')

writer.close()
print('Finished Training')

