import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import Models

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchSize = 256

transformsTrain = torchvision.transforms.Compose([
    transforms.RandomCrop(size=32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
])

transformsTest = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
])

trainSet = datasets.CIFAR10(root='datasets', download=False, train=True, transform=transformsTrain)
testSet = datasets.CIFAR10(root='datasets', download=False, train=False, transform=transformsTest)

classes = trainSet.classes
print(f"train: {len(trainSet)}, test: {len(testSet)}")
print(classes)
print(f"Number of classes {len(classes)}")

trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=True)


def dataDisplay(name="cifar10"):
    data = iter(trainLoader)
    images, labels = data.next()
    fig = plt.figure(figsize=(70, 25))
    imgNumber = 50
    for index in range(1, imgNumber + 1):
        plt.subplot(5, 10, index)
        plt.axis('off')
        plt.imshow((np.transpose(images[index].numpy(), (1, 2, 0)) * 255).astype(np.uint8))
        plt.title(classes[labels.numpy()[index]], fontsize=30)
    plt.show()
    fig.savefig(name, dpi=fig.dpi)


def trainModel(string="cifar-10", modelName="model1.pth", epochs=100, lr=1e-3, model=Models.Cifar10().to(device)):
    path = f"models/{string}/{modelName}"
    writer = SummaryWriter(f"runs/{string}/{modelName.partition('.')[0]}")
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0

    # Checkpoint
    # checkpoint = torch.load('models/checkpoints/cp.pth')
    # model.load_state_dict(checkpoint['model_state'])
    # optimizer.load_state_dict(checkpoint['optimizer_state'])
    # epoch = checkpoint['epoch'] + 1
    # loss = checkpoint['loss']

    timeStart = time()

    for e in range(epoch, epochs):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=True)
        for idx, (images, labels) in loop:
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            labels = labels.to(device)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), (e * int(len(trainSet) / batchSize)) + idx)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = 100. * (correct / len(predictions))
            loop.set_description(f"Epoch [{e}/{epochs}")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
            writer.add_scalar('acc', accuracy, (e * int(len(trainSet) / batchSize)) + idx)
        else:
            torch.save({
                'epoch': e,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss_fn
            }, f'models/{string}/checkpoints/cp' + str(e) + '.pth')

    print("Trenirali smo (u minutama):", (time() - timeStart) / 60)
    torch.save(model, path)
    return model


imgLables = []
predLables = []


def test(string="models/cifar-10/model1.pth"):
    model = torch.load(string)
    correctPred = {classname: 0 for classname in classes}
    totalPred = {classname: 0 for classname in classes}
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in testLoader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            for label, prediction in zip(y, predictions):
                if label != prediction:
                    imgLables.append(label.cpu().numpy())
                    predLables.append(prediction.cpu().numpy())
                else:
                    correctPred[classes[label]] += 1
                totalPred[classes[label]] += 1

        for classname, correct_count in correctPred.items():
            accuracy = 100 * float(correct_count) / totalPred[classname]
            print(f"{classname}: {accuracy:.2f}%")
        print(f'{num_correct}/{num_samples} -> {float(num_correct) / float(num_samples) * 100:.2f}%\n')


# dataDisplay()
trainModel()
test()
