import torch
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from models.senet import *
from models.senet2 import *
from models.inception_resnet import *
from models.resnet import *
import numpy as np
from statistics import mean
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--checkpoint', type=int, default=0)
parser.add_argument('--mixed', type=int, default=0)
params = parser.parse_args()

num_epochs = params.epochs
batch_size = params.batch
checkpoint = params.checkpoint
mixed = params.mixed

def my_collate(batch):
    data = [item[0] for item in batch]
    # data = torch.LongTensor(data)
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

mixed_dict = {
    0:[0,11],
    1:[0,68],
    2:[10,11],
    3:[11,57],
    4:[11,68],
    5:[11,99],
    6:[11,102],
    7:[57,113,114,115]
}

train_root = 'train/'
val_root = 'val/'

mixed_train_root = 'mix_train/'
mixed_val_root = 'mix_val/'

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)

train_mixed = datasets.ImageFolder(root=mixed_train_root, transform=base_transform)
val_mixed = datasets.ImageFolder(root=mixed_val_root, transform=base_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)

mixed_train_loader = torch.utils.data.DataLoader(
    train_mixed, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)
mixed_val_loader = torch.utils.data.DataLoader(
    val_mixed, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# model = se_resnet18(120).to(device)
# model = InceptionResV2().to(device)
model = ResNet18(120)
if checkpoint > 0:
    if mixed == 0:
        model.load_state_dict(torch.load(f'checkpoint_std/model.{checkpoint}'))
    else:
        model.load_state_dict(torch.load(f'checkpoint/model.{checkpoint}'))

criterion = nn.MultiLabelMarginLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay = 5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

losses_f = open('logs/losses.txt','a')
acc_f = open('logs/acc.txt', 'a')

def train_normal(epoch):
    running_loss = 0.0
    top5_acc,top1_acc = [],[]
    for batch_num, (inputs, labels) in enumerate(train_loader,1):
        model.train()
        multi_labels = []
        for label in labels:
            ml = [-1]*120
            ml[0] = label
            multi_labels.append(torch.LongTensor(ml))
        inputs,labels = torch.stack(inputs).to(device), torch.stack(multi_labels).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels).to(device)
        # print (loss)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        model.eval()
        top5 = torch.topk(outputs,k=5)[1]
        top1 = torch.topk(outputs,k=1)[1]

        top5_acc.append(mean([int(label[0].item() in top5[i]) for i,label in enumerate(labels)]))
        top1_acc.append(mean([int(label[0].item() in top1[i]) for i,label in enumerate(labels)]))

        # print (running_loss, mean(top5_acc))
        if batch_num % 50 == 0:
            print (epoch, batch_num, running_loss, mean(top5_acc), mean(top1_acc))
            losses_f.write(f'{epoch} : {batch_num} : {running_loss} : {mean(top1_acc)} : {mean(top5_acc)}\n')
            running_loss = 0.0

        gc.collect()
        torch.cuda.empty_cache()
    return top5_acc, top1_acc

def train_mixed(epoch):
    running_loss = 0.0
    top5_acc,top1_acc = [],[]
    for batch_num, (inputs, labels) in enumerate(mixed_train_loader,1):
        model.train()
        multi_labels = []
        for label in labels:
            ml = [-1]*120
            for i,idx in enumerate(mixed_dict[label.item()]):
                ml[i] = idx
            multi_labels.append(torch.LongTensor(ml))
        inputs,labels = torch.stack(inputs).to(device), torch.stack(multi_labels).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels).to(device)
        # print (loss)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        model.eval()
        top5 = torch.topk(outputs,k=5)[1]
        top1 = torch.topk(outputs,k=1)[1]

        top5_acc.append(mean([int(label[0].item() in top5[i] or label[1].item() in top5[i]) for i,label in enumerate(labels)]))
        top1_acc.append(mean([int(label[0].item() in top1[i] or label[1].item() in top1[i]) for i,label in enumerate(labels)]))
        # print (running_loss, mean(top5_acc))
        if batch_num % 50 == 0:
            print (epoch, batch_num, running_loss, mean(top5_acc), mean(top1_acc))
            losses_f.write(f'{epoch} : {batch_num} : {running_loss} : {mean(top1_acc)} : {mean(top5_acc)}\n')
            running_loss = 0.0

        gc.collect()
        torch.cuda.empty_cache()
    return top5_acc, top1_acc

def val_normal(epoch):
    top5_val, top1_val, val_loss = [],[],[]
    model.eval()
    for batch_num, (inputs, labels) in enumerate(val_loader,1):
        multi_labels = []
        for label in labels:
            ml = [-1]*120
            ml[0] = label
            multi_labels.append(torch.LongTensor(ml))
        inputs,labels = torch.stack(inputs).to(device), torch.stack(multi_labels).to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=1)[1]

        top5_val.append(mean([int(label[0].item() in top5[i]) for i,label in enumerate(labels)]))
        top1_val.append(mean([int(label[0].item() in top1[i]) for i,label in enumerate(labels)]))
        val_loss.append(criterion(outputs, labels).to(device).item())
    return top5_val, top1_val, val_loss

def val_mixed(epoch):
    top5_val, top1_val, val_loss = [],[],[]
    model.eval()
    for batch_num, (inputs, labels) in enumerate(mixed_train_loader,1):
        multi_labels = []
        for label in labels:
            ml = [-1]*120
            for i,idx in enumerate(mixed_dict[label.item()]):
                ml[i] = idx
            multi_labels.append(torch.LongTensor(ml))
        inputs,labels = torch.stack(inputs).to(device), torch.stack(multi_labels).to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=1)[1]

        top5_val.append(mean([int(label[0].item() in top5[i] or label[1].item() in top5[i]) for i,label in enumerate(labels)]))
        top1_val.append(mean([int(label[0].item() in top1[i] or label[1].item() in top1[i]) for i,label in enumerate(labels)]))
        val_loss.append(criterion(outputs, labels).to(device).item())
    return top5_val, top1_val, val_loss

def run(epoch):
    top5_acc,top1_acc = [],[]
    if mixed > 0:
        top5_acc, top1_acc = train_mixed(epoch)
        top5_val, top1_val, val_loss = val_mixed(epoch)
    else:
        top5_acc, top1_acc = train_normal(epoch)
        top5_val, top1_val, val_loss = val_normal(epoch)

    print('VAL:', epoch, mean(top1_val), mean(top5_val))
    acc_f.write(f'{epoch} : {mean(top1_acc)} : {mean(top5_acc)} : {mean(top1_val)} : {mean(top5_val)}\n')
    scheduler.step(mean(val_loss))

    torch.save(model.state_dict(), f'checkpoint/model.{epoch}')
    gc.collect()
    torch.cuda.empty_cache()

for epoch in range(checkpoint+1,checkpoint+num_epochs+1):
    run(epoch)
