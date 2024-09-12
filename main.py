# Importing Libraries
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm


# Custom dataset class to read VOC2007 annotations
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Paths to images and annotations
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.annotation_dir = os.path.join(root, 'Annotations')

        # List image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

        self.class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        # Initialize an empty list to hold image paths, bounding boxes, and labels
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # Iterate over image files and load corresponding annotations
        for image_file in self.image_files:
            img_path = os.path.join(self.image_dir, image_file)
            ann_path = os.path.join(self.annotation_dir, image_file.replace('.jpg', '.xml'))
            # Parse the annotation XML
            tree = ET.parse(ann_path)
            root = tree.getroot()
            # Extract bounding boxes and labels
            for obj in root.findall('object'):
                name = obj.find('name').text
                label = self.class_names.index(name)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                # Save the sample with image path, bounding box, and label
                self.samples.append((img_path, (xmin, ymin, xmax, ymax), label))

    def __getitem__(self, index):
        img_path, box, label = self.samples[index]
        # Load the image
        img = Image.open(img_path).convert('RGB')

        # TODO: 补全代码 填在下方

        # 补全内容: 跟据给定框的参数裁剪出子图用于分类 注意box中元素的对应关系
        # cropped_img ...

        # TODO: 补全代码 填在上方

        # Apply the transform if provided
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)

        return cropped_img, label

    def __len__(self):
        return len(self.samples)


# Define the model, here we take resnet-18 as an example
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        DROPOUT = 0.1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)

# TODO: 调整模型的架构 分析讨论其他网络设计的表现 注意ResNet内部也需要进行调整 也可直接使用timm中的模型架构。
def design_model(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):
            
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # TODO: 补全代码 填在下方

        # 补全内容: optimizer的操作 获取模型输出 loss设计与计算 反向传播
        # optimizer ...
        # y_pred ...
        # loss ...

        # TODO: 补全代码 填在上方

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)


def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = []):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)

            # TODO: 补全代码 填在下方

            # 补全内容: 获取模型输出 loss计算
            # output ...
            # test_loss ...

            # TODO: 补全代码 填在上方

            pred = output.argmax(dim=1, keepdim=True)
            for d,i,j in zip(data, pred, target):
                if i != j:
                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()])
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    
    test_acc.append(100. * correct / len(test_dataloader.dataset))

def main():
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(device)
    resolution = 32
    
    #prepare datasets and transforms
    train_transforms = transforms.Compose([ #resises the image so it can be perfect for our model.
        
            # TODO: 补全代码 填在下方

            # 设计针对训练数据集的图像增强以提高模型性能

            # TODO: 补全代码 填在上方

            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
            # TODO: Normalize值来自于Cifar数据集 可通过统计代码获得更适合Voc2007子图的值 test_transforms采用一样的值

            ])
    test_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])

    # Adjust the root directory according to your VOC2007 data path
    data_dir = './voc2007'
    # Load the complete training dataset
    trainval_set = VOCDataset(os.path.join(data_dir, 'trainval'))

    # Define the split sizes: 80% training, 20% validation
    train_size = int(0.8 * len(trainval_set))
    val_size = len(trainval_set) - train_size

    # Split the dataset into training and validation sets
    trainset, valset = random_split(trainval_set, [train_size, val_size])

    # Manually apply the transforms to the training and validation datasets
    trainset.dataset.transform = train_transforms
    valset.dataset.transform = test_transforms

    # Create DataLoader objects
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Model creation
    num_classes = len(trainval_set.class_names)
    model = design_model(num_classes).to(device)
    summary(model, input_size=(3, resolution, resolution))

    # Training the model

    # TODO: 更改optimizer和scheduler的设置以获得最优参数
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    train_acc = []
    train_losses = []
    val_acc = []
    val_losses = []
    model_path = './checkpoints'
    os.makedirs(model_path, exist_ok=True)

    EPOCHS = 40
    best_val_acc = 0.0  # Variable to keep track of the best val accuracy

    for i in range(EPOCHS):
        print(f'EPOCH : {i+1}/{EPOCHS}')
        
        # Train the model for one epoch

        # TODO: 补全model_training里的代码
        model_training(model, device, trainloader, optimizer, train_acc, train_losses)
        
        # Update the learning rate scheduler
        scheduler.step(train_losses[-1])
        
        # Val the model after training

        # TODO: 补全model_testing里的代码
        model_testing(model, device, valloader, val_acc, val_losses)
        
        # Save the latest model
        torch.save(model.state_dict(), os.path.join(model_path, 'latest_model.pth'))
        
        # Save the best model based on test accuracy
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            print(f"Best model saved with test accuracy: {best_val_acc:.4f}")


    fig, axs = plt.subplots(2,2, figsize=(25,20))
    axs[0,0].set_title('Train Losses')
    axs[0,1].set_title(f'Training Accuracy (Max: {max(train_acc):.2f})')
    axs[1,0].set_title('Val Losses')
    axs[1, 1].set_title(f'Val Accuracy (Max: {max(val_acc):.2f})')
    axs[0,0].plot(train_losses)
    axs[0,1].plot(train_acc)
    axs[1,0].plot(val_losses)
    axs[1,1].plot(val_acc)

    # draw results
    plt.savefig('curves.png')  


if __name__ == '__main__':
    main()