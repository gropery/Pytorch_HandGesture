import cv2 as cv
import torchvision
from torchvision import transforms
from model import *

# 转换图片numpy array格式为tensor类型，且归一化数据
tensor_trans = transforms.ToTensor()

# 加载训练好的网络
# net = MyNet()
# net = torchvision.models.alexnet(weights=None, num_classes=10)
# net = torchvision.models.vgg16(weights=None, num_classes=10)
net = torchvision.models.resnet50(weights=None, num_classes=10)
net.load_state_dict(torch.load('./pths/gesture_30.pth', map_location=torch.device('cpu')))

# 从图片中读取图像分类 ##############################################
img_path = 'datasets/Sign-Language-Digits-Dataset/Examples/example_6.JPG'
# opencv 读取图片得到 numpy.ndarray (H x W x C)格式数据 channel为BGR
image = cv.imread(img_path)
# 改变图片尺寸以使得符合网络输入
if not image.shape == (100, 100, 3):
    image = cv.resize(image, (100, 100))
# 转换图片numpy array格式为tensor类型
image = tensor_trans(image)
print(image.shape)

# 输入图片使用网络进行测试
image = torch.reshape(image, (1, 3, 100, 100))
net.eval()
with torch.no_grad():
    output = net(image)

print(output)
print(output.argmax(1))
print(int(output.argmax(1)))

# 从摄像机中读取图像分类 ##############################################
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 开始预测
    im = cv.resize(frame, (100, 100))
    im = tensor_trans(im)
    im = torch.reshape(im, (1, 3, 100, 100))
    output = net(im)
    print(str(int(output.argmax(1))))

    # 添加文本
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, str(int(output.argmax(1))), (10, 50), font, 1, (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()

