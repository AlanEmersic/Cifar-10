import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=inputDim, out_features=512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.dropout4(x)
        x = self.fc4(x)
        return x

    def calculateDimensions(self, size=32):
        batchData = torch.zeros((1, 3, size, size))
        batchData = self.conv1(batchData)
        # batchData = self.pool1(batchData)
        batchData = self.conv2(batchData)
        batchData = self.pool2(batchData)
        batchData = self.conv3(batchData)
        # batchData = self.pool3(batchData)
        batchData = self.conv4(batchData)
        batchData = self.pool4(batchData)

        return int(np.product(batchData.size()))


class Cifar3(nn.Module):
    def __init__(self):
        super(Cifar3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=inputDim, out_features=512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.dropout4(x)
        x = self.fc4(x)
        return x

    def calculateDimensions(self, size=32):
        batchData = torch.zeros((1, 3, size, size))
        batchData = self.conv1(batchData)
        # batchData = self.pool1(batchData)
        batchData = self.conv2(batchData)
        batchData = self.pool2(batchData)
        batchData = self.conv3(batchData)
        batchData = self.pool3(batchData)
        batchData = self.conv4(batchData)
        batchData = self.pool4(batchData)

        return int(np.product(batchData.size()))


class Cifar4(nn.Module):
    def __init__(self):
        super(Cifar4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=inputDim, out_features=512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.dropout4(x)
        x = self.fc4(x)
        return x

    def calculateDimensions(self, size=32):
        batchData = torch.zeros((1, 3, size, size))
        batchData = self.conv1(batchData)
        # batchData = self.pool1(batchData)
        batchData = self.conv2(batchData)
        batchData = self.pool2(batchData)
        batchData = self.conv3(batchData)
        batchData = self.pool3(batchData)
        batchData = self.conv4(batchData)
        batchData = self.pool4(batchData)

        return int(np.product(batchData.size()))


class Cifar5(nn.Module):
    def __init__(self):
        super(Cifar5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout6 = nn.Dropout(p=0.2)

        # Linear
        inputDim = self.calculateDimensions()
        # self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=inputDim, out_features=10)

        # self.dropout4 = nn.Dropout(p=0.5)
        # self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool6(x)
        x = self.dropout6(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def calculateDimensions(self, size=32):
        batchData = torch.zeros((1, 3, size, size))
        batchData = self.conv1(batchData)
        batchData = self.conv2(batchData)
        batchData = self.pool2(batchData)

        batchData = self.conv3(batchData)
        batchData = self.conv4(batchData)
        batchData = self.pool4(batchData)

        batchData = self.conv5(batchData)
        batchData = self.conv6(batchData)
        batchData = self.pool6(batchData)

        return int(np.product(batchData.size()))


class Cifar6(nn.Module):
    def __init__(self):
        super(Cifar6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.4)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout(p=0.4)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout6 = nn.Dropout(p=0.4)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout8 = nn.Dropout(p=0.4)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.dropout9 = nn.Dropout(p=0.4)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.dropout11 = nn.Dropout(p=0.4)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.dropout12 = nn.Dropout(p=0.4)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool13 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        self.dropout14 = nn.Dropout(p=0.5)
        self.fc14 = nn.Linear(in_features=inputDim, out_features=512)
        self.bn14 = nn.BatchNorm1d(512)

        self.dropout15 = nn.Dropout(p=0.5)
        self.fc15 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.pool7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.pool10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.dropout11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.dropout12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.pool13(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.dropout14(x)
        x = self.fc14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.dropout15(x)
        x = self.fc15(x)
        return x

    def calculateDimensions(self, size=32):
        batchData = torch.zeros((1, 3, size, size))
        batchData = self.conv1(batchData)
        batchData = self.dropout1(batchData)

        batchData = self.conv2(batchData)
        batchData = self.pool2(batchData)

        batchData = self.conv3(batchData)
        batchData = self.dropout3(batchData)

        batchData = self.conv4(batchData)
        batchData = self.pool4(batchData)

        batchData = self.conv5(batchData)
        batchData = self.dropout5(batchData)

        batchData = self.conv6(batchData)
        batchData = self.dropout6(batchData)

        batchData = self.conv7(batchData)
        batchData = self.pool7(batchData)

        batchData = self.conv8(batchData)
        batchData = self.dropout8(batchData)

        batchData = self.conv9(batchData)
        batchData = self.dropout9(batchData)

        batchData = self.conv10(batchData)
        batchData = self.pool10(batchData)

        batchData = self.conv11(batchData)
        batchData = self.dropout11(batchData)

        batchData = self.conv12(batchData)
        batchData = self.dropout12(batchData)

        batchData = self.conv13(batchData)
        batchData = self.pool13(batchData)

        return int(np.product(batchData.size()))


class Cifar7(nn.Module):
    def __init__(self):
        super(Cifar7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(512)

        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.pool16 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        # self.dropout14 = nn.Dropout(p=0.5)
        self.fc17 = nn.Linear(in_features=inputDim, out_features=4096)
        self.bn17 = nn.BatchNorm1d(4096)

        # self.dropout15 = nn.Dropout(p=0.5)
        self.fc18 = nn.Linear(in_features=4096, out_features=1000)
        self.bn18 = nn.BatchNorm1d(1000)

        self.fc19 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.dropout5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        # x = self.dropout6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        # x = self.pool7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        # x = self.dropout8(x)
        x = self.pool8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        # x = self.dropout9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        # x = self.pool10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        # x = self.dropout11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        # x = self.dropout12(x)
        x = self.pool12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        # x = self.pool13(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)
        x = self.pool16(x)

        # Linear
        x = x.view(x.size(0), -1)
        # x = self.dropout14(x)
        x = self.fc17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.fc18(x)
        x = self.bn18(x)
        x = F.relu(x)

        # x = self.dropout15(x)
        x = self.fc19(x)
        return x

    def calculateDimensions(self, size=32):
        x = torch.zeros((1, 3, size, size))
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.pool4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        x = self.conv7(x)

        x = self.conv8(x)

        x = self.pool8(x)

        x = self.conv9(x)

        x = self.conv10(x)

        x = self.conv11(x)

        x = self.conv12(x)
        x = self.pool12(x)

        x = self.conv13(x)

        x = self.conv14(x)

        x = self.conv15(x)

        x = self.conv16(x)

        x = self.pool16(x)

        return int(np.product(x.size()))


class Cifar8(nn.Module):
    def __init__(self):
        super(Cifar8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(512)

        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(512)

        self.conv17 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(1024)

        self.conv18 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(1024)

        self.conv18 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn19 = nn.BatchNorm2d(1024)

        self.conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn20 = nn.BatchNorm2d(1024)
        self.pool20 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear
        inputDim = self.calculateDimensions()
        # self.dropout14 = nn.Dropout(p=0.5)
        self.fc21 = nn.Linear(in_features=inputDim, out_features=4096)
        self.bn21 = nn.BatchNorm1d(4096)

        # self.dropout15 = nn.Dropout(p=0.5)
        self.fc22 = nn.Linear(in_features=4096, out_features=4096)
        self.bn22 = nn.BatchNorm1d(4096)

        self.fc23 = nn.Linear(in_features=4096, out_features=512)
        self.bn23 = nn.BatchNorm1d(512)

        self.fc24 = nn.Linear(in_features=512, out_features=512)
        self.bn24 = nn.BatchNorm1d(512)

        self.fc25 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.pool8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.pool12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = F.relu(x)

        x = self.conv19(x)
        x = self.bn19(x)
        x = F.relu(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = F.relu(x)
        x = self.pool20(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.fc21(x)
        x = self.bn21(x)
        x = F.relu(x)

        x = self.fc22(x)
        x = self.bn22(x)
        x = F.relu(x)

        x = self.fc23(x)
        x = self.bn23(x)
        x = F.relu(x)

        x = self.fc24(x)
        x = self.bn24(x)
        x = F.relu(x)

        x = self.fc25(x)
        return x

    def calculateDimensions(self, size=32):
        x = torch.zeros((1, 3, size, size))
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.pool4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        x = self.conv7(x)

        x = self.conv8(x)

        x = self.pool8(x)

        x = self.conv9(x)

        x = self.conv10(x)

        x = self.conv11(x)

        x = self.conv12(x)
        x = self.pool12(x)

        x = self.conv13(x)

        x = self.conv14(x)

        x = self.conv15(x)

        x = self.conv16(x)

        x = self.conv17(x)

        x = self.conv18(x)

        x = self.conv19(x)

        x = self.conv20(x)
        x = self.pool20(x)

        return int(np.product(x.size()))


class Cifar9(nn.Module):
    def __init__(self):
        super(Cifar9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 6 64,64,3
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn7 = nn.BatchNorm2d(64)

        # 8 128,128,2
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=2)
        self.bn8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn10 = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn11 = nn.BatchNorm2d(128)

        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn12 = nn.BatchNorm2d(128)

        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn13 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn14 = nn.BatchNorm2d(128)

        self.conv15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn15 = nn.BatchNorm2d(128)

        # 12 256,256,2
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=2)
        self.bn16 = nn.BatchNorm2d(256)

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn17 = nn.BatchNorm2d(256)

        self.conv18 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv19 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn19 = nn.BatchNorm2d(256)

        self.conv20 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn20 = nn.BatchNorm2d(256)

        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn21 = nn.BatchNorm2d(256)

        self.conv22 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn22 = nn.BatchNorm2d(256)

        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn23 = nn.BatchNorm2d(256)

        self.conv24 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn24 = nn.BatchNorm2d(256)

        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn25 = nn.BatchNorm2d(256)

        self.conv26 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn26 = nn.BatchNorm2d(256)

        self.conv27 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn27 = nn.BatchNorm2d(256)

        # 6 512,512,2
        self.conv28 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding=2)
        self.bn28 = nn.BatchNorm2d(512)

        self.conv29 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn29 = nn.BatchNorm2d(512)

        self.conv30 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn30 = nn.BatchNorm2d(512)

        self.conv31 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn31 = nn.BatchNorm2d(512)

        self.conv32 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn32 = nn.BatchNorm2d(512)

        self.conv33 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn33 = nn.BatchNorm2d(512)
        self.pool33 = nn.AvgPool2d(2, 2)

        # Linear
        inputDim = self.calculateDimensions()
        self.fc34 = nn.Linear(in_features=inputDim, out_features=1000)
        self.bn34 = nn.BatchNorm1d(1000)

        self.fc35 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = F.relu(x)

        x = self.conv19(x)
        x = self.bn19(x)
        x = F.relu(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = F.relu(x)

        x = self.conv21(x)
        x = self.bn21(x)
        x = F.relu(x)

        x = self.conv22(x)
        x = self.bn22(x)
        x = F.relu(x)

        x = self.conv23(x)
        x = self.bn23(x)
        x = F.relu(x)

        x = self.conv24(x)
        x = self.bn24(x)
        x = F.relu(x)

        x = self.conv25(x)
        x = self.bn25(x)
        x = F.relu(x)

        x = self.conv26(x)
        x = self.bn26(x)
        x = F.relu(x)

        x = self.conv27(x)
        x = self.bn27(x)
        x = F.relu(x)

        x = self.conv28(x)
        x = self.bn28(x)
        x = F.relu(x)

        x = self.conv29(x)
        x = self.bn29(x)
        x = F.relu(x)

        x = self.conv30(x)
        x = self.bn30(x)
        x = F.relu(x)

        x = self.conv31(x)
        x = self.bn31(x)
        x = F.relu(x)

        x = self.conv32(x)
        x = self.bn32(x)
        x = F.relu(x)

        x = self.conv33(x)
        x = self.bn33(x)
        x = F.relu(x)
        x = self.pool33(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.fc34(x)
        x = self.bn34(x)
        x = F.relu(x)

        x = self.fc35(x)
        return x

    def calculateDimensions(self, size=32):
        x = torch.zeros((1, 3, size, size))
        x = self.conv1(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)
        x = self.conv27(x)

        x = self.conv28(x)
        x = self.conv29(x)
        x = self.conv30(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pool33(x)

        return int(np.product(x.size()))

class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        # 6 64,64,3
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn7 = nn.BatchNorm2d(64)

        # 8 128,128,2
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=2)
        self.bn8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn10 = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn11 = nn.BatchNorm2d(128)

        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn12 = nn.BatchNorm2d(128)

        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn13 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn14 = nn.BatchNorm2d(128)

        self.conv15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn15 = nn.BatchNorm2d(128)

        # 5 256,256,2
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=2)
        self.bn16 = nn.BatchNorm2d(256)

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn17 = nn.BatchNorm2d(256)

        self.conv18 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv19 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn19 = nn.BatchNorm2d(256)

        self.conv20 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.bn20 = nn.BatchNorm2d(256)


        # 6 512,512,2
        self.conv28 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding=2)
        self.bn28 = nn.BatchNorm2d(512)

        self.conv29 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn29 = nn.BatchNorm2d(512)

        self.conv30 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn30 = nn.BatchNorm2d(512)

        self.conv31 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn31 = nn.BatchNorm2d(512)

        self.conv32 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn32 = nn.BatchNorm2d(512)

        self.conv33 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
        self.bn33 = nn.BatchNorm2d(512)
        self.pool33 = nn.AvgPool2d(2, 2)

        # Linear
        inputDim = self.calculateDimensions()
        self.fc34 = nn.Linear(in_features=inputDim, out_features=512)
        self.bn34 = nn.BatchNorm1d(512)

        self.fc35 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = F.relu(x)

        x = self.conv19(x)
        x = self.bn19(x)
        x = F.relu(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = F.relu(x)

        # x = self.conv21(x)
        # x = self.bn21(x)
        # x = F.relu(x)
        #
        # x = self.conv22(x)
        # x = self.bn22(x)
        # x = F.relu(x)
        #
        # x = self.conv23(x)
        # x = self.bn23(x)
        # x = F.relu(x)
        #
        # x = self.conv24(x)
        # x = self.bn24(x)
        # x = F.relu(x)
        #
        # x = self.conv25(x)
        # x = self.bn25(x)
        # x = F.relu(x)
        #
        # x = self.conv26(x)
        # x = self.bn26(x)
        # x = F.relu(x)
        #
        # x = self.conv27(x)
        # x = self.bn27(x)
        # x = F.relu(x)

        x = self.conv28(x)
        x = self.bn28(x)
        x = F.relu(x)

        x = self.conv29(x)
        x = self.bn29(x)
        x = F.relu(x)

        x = self.conv30(x)
        x = self.bn30(x)
        x = F.relu(x)

        x = self.conv31(x)
        x = self.bn31(x)
        x = F.relu(x)

        x = self.conv32(x)
        x = self.bn32(x)
        x = F.relu(x)

        x = self.conv33(x)
        x = self.bn33(x)
        x = F.relu(x)
        x = self.pool33(x)

        # Linear
        x = x.view(x.size(0), -1)
        x = self.fc34(x)
        x = self.bn34(x)
        x = F.relu(x)

        x = self.fc35(x)
        return x

    def calculateDimensions(self, size=32):
        x = torch.zeros((1, 3, size, size))
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)

        x = self.conv28(x)
        x = self.conv29(x)
        x = self.conv30(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pool33(x)

        return int(np.product(x.size()))