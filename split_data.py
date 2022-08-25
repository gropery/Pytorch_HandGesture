import glob

# 数据集根目录
root_dir = 'datasets/Sign-Language-Digits-Dataset/Dataset'

# 根目录下的分类目录
dirs_collect = glob.glob(root_dir+'/*')

# 每个具体类文件夹中的文件按8:2分为训练和测试集
count = 0
WHOLE_SPLIT = 10
TRAIN_SPLIT = 8
TEST_SPLIT = WHOLE_SPLIT - TRAIN_SPLIT

# 保存训练和测试集的地址，和类型文件
train_file = open('train.txt', 'w')
test_file = open('test.txt', 'w')

for ges in dirs_collect:               # 遍历每个分类文件夹名称，同时作为labels
    print(ges)
    label = ges.split('\\')[-1]
    print(label)
    images_list = glob.glob(ges+'/*')
    # print(images_list)
    for img_index in images_list:      # 遍历每个具体分类文件夹下的文件地址，按照比例分别写入训练和测试集
        count = count + 1
        if count <= TRAIN_SPLIT:
            # WRITE TRIAN
            train_file.write(img_index+','+label+'\n')
        elif count <= WHOLE_SPLIT:
            # WRITE TEST
            test_file.write(img_index+','+label+'\n')
        if count == WHOLE_SPLIT:
            count = 0

train_file.close()
test_file.close()
