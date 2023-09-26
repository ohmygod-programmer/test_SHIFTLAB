import os
import io
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch import nn
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(35 * 35 * 60, 1000)
        self.fc2_1 = nn.Linear(1000, 3)
        self.fc2_2 = nn.Linear(1000, 4)
        self.fc2_3 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out1 = self.fc2_1(out)
        out2 = self.fc2_2(out)
        out3 = self.fc2_3(out)
        return out1, out2, out3

preprocess = T.Compose([
   T.Resize(512),
   T.CenterCrop(444),
   T.ToTensor(),
   T.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])

#Датасет, позволяющий хранить несколько классов для одного объекта
class MultiLabelDataset(VisionDataset):
    def __init__(self, root, names_df: pd.DataFrame, labels_df: pd.DataFrame, transform=None, target_transform=None, labels_num=1, labels_sizes=[1]):
        super(MultiLabelDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        if (len(names_df) != len(labels_df)):
            raise Exception
        self.len = len(names_df)
        self.names_df = names_df
        self.labels_df = labels_df
        self.labels_num = labels_num
        self.labels_sizes = labels_sizes

    def __getitem__(self, index):
        path = self.names_df['path'][index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        # Get the labels for the corresponding image
        labels = self.labels_df.iloc[index].values
        if (self.labels_num > 1):
            if (self.labels_num == len(self.labels_sizes)):
                new_labels = []
                start = 0
                for i in range(self.labels_num):
                    new_labels.append(labels[start:start+self.labels_sizes[i]])
                    start += self.labels_sizes[i]
                return img, new_labels
            else:
                raise Exception
        return img, labels

    def __len__(self) -> int:
        return self.len


#Заполнение датасета
paths_df = pd.DataFrame(columns=['path'])
labels_df = pd.DataFrame(columns=['sharp', 'blurred', 'torched', '0d', '90d', '180d', '270d', 'rgb', 'bgr'], dtype=float)
index = 0
for i in os.walk('dataset'):
    for file in  i[2]:
        if (file.lower().endswith(('.jpg', '.jpeg'))):
            pathdf = pd.DataFrame({'path': os.path.join(i[0], file)}, index=[index])
            if (str(file).find('sharp') != -1):
                sbt = (1, 0, 0)
            elif (str(file).find('blurred') != -1):
                sbt = (0, 1, 0)
            else:
                sbt = (0, 0, 1)

            if (str(file).find('90d') != -1):
                degr = (0, 1, 0, 0)
            elif (str(file).find('180d') != -1):
                degr = (0, 0, 1, 0)
            elif (str(file).find('270d') != -1):
                degr = (0, 0, 0, 1)
            else:
                degr = (1, 0, 0, 0)
            if (str(file).find('rgb') != -1):
                mod = (1, 0)
            else:
                mod = (0, 1)
            labeldf = pd.DataFrame(data=[sbt + degr + mod], columns=['sharp', 'blurred', 'torched', '0d', '90d', '180d', '270d', 'rgb', 'bgr'], index=[index])
            paths_df = pd.concat([paths_df, pathdf])
            labels_df = pd.concat([labels_df, labeldf])
            index += 1

dataset = MultiLabelDataset(root='dataset', names_df=paths_df, labels_df=labels_df, transform=preprocess, labels_num=3, labels_sizes=[3, 4, 2])

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=4)



num_epochs = 7
learning_rate = 0.001
labels_num = 3

model = ConvNet()

# Loss and optimizer
criterions = [nn.CrossEntropyLoss() for i in range(labels_num)]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Прямой запуск
        outs = model(images)
        loss = torch.Tensor([0])
        for j, out in enumerate(outs):
            loss += criterions[j](out, labels[j])

        loss_list.append(loss.item())

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

all_pred = [torch.Tensor() for i in range(labels_num)]
all_l = [torch.Tensor() for i in range(labels_num)]
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        for j, out in enumerate(outputs):
            _, predicted = torch.max(out.data, 1)
            _, l = torch.max(labels[j].data, 1)
            all_pred[j] = torch.cat((all_pred[j], predicted), dim=0)
            all_l[j] = torch.cat((all_l[j], l), dim=0)
for i in range(labels_num):
    print('Accuracy:',accuracy_score(all_pred[i], all_l[i]))
    print('Confusion matrix:\n',confusion_matrix(all_pred[i], all_l[i]))
    print('F1_score:',f1_score(all_pred[i], all_l[i], average=None))