import torch
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from models.senet import *
from statistics import mean
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--checkpoint', type=int, default=0)
params = parser.parse_args()

num_epochs = params.epochs
batch_size = params.batch
checkpoint = params.checkpoint

def my_collate(batch):
    data = [item[0] for item in batch]
    # data = torch.LongTensor(data)
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

train_root = 'train/'
val_root = 'val/'
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])
train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = se_resnet18(120).to(device)
if checkpoint > 0:
    model.load_state_dict(torch.load(f'checkpoint/model.{checkpoint}'))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay = 5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

losses_f = open('logs/losses.txt','a')
acc_f = open('logs/acc.txt', 'a')

def train_normal(epoch):
    running_loss = 0.0
    top5_acc,top1_acc = [],[]
    for batch_num, (inputs, labels) in enumerate(train_loader,1):
        # print('Batch: ',inputs[0],labels)
        model.train()
        # inputs,labels = torch.stack(inputs).to(device), labels.to(device)
        inputs,labels = torch.stack(inputs).to(device), labels.to(device)
        # inputs,labels = torch.unsqueeze(inputs[0],0).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels).to(device)
        # print (loss)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        model.eval()
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=2)[1]
        top5_acc.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
        top5_acc.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
        top1_acc.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))
        top1_acc.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))
        # print (top5_acc, top1_acc)

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
    for batch_num, (inputs, labels) in enumerate(train_loader,1):
        # print('Batch: ',inputs[0],labels)
        mixed_input = []
        mixed_label = []
        for i in range (len(inputs)-1):
            double = torch.stack(inputs[i:i+2])
            mixed_input.append(torch.mean(double,0))
            mixed_label.append(labels[i:i+2])
        model.train()
        # inputs,labels = torch.stack(inputs).to(device), labels.to(device)
        inputs,labels = torch.stack(mixed_input).to(device), torch.stack(mixed_label).to(device)
        # inputs,labels = torch.unsqueeze(inputs[0],0).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = sum(criterion(outputs, labels[:,i]).to(device) for i in range(2))
        # print (loss)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        model.eval()
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=2)[1]
        top5_acc.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels[:,0])]))
        top5_acc.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels[:,1])]))
        top1_acc.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels[:,0])]))
        top1_acc.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels[:,1])]))
        # print (top5_acc, top1_acc)

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
    for batch_num, (inputs, labels) in enumerate(val_loader):
        inputs,labels = torch.stack(inputs).to(device), labels.to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=2)[1]
        top5_val.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
        top5_val.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
        top1_val.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))
        top1_val.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))
        val_loss.append(sum(criterion(outputs, labels).to(device) for i in range(2)).item())
    return top5_val, top1_val, val_loss

def val_mixed(epoch):
    top5_val, top1_val, val_loss = [],[],[]
    model.eval()
    for batch_num, (inputs, labels) in enumerate(val_loader):
        mixed_input = []
        mixed_label = []
        for i in range (len(inputs)-1):
            double = torch.stack(inputs[i:i+2])
            mixed_input.append(torch.mean(double,0))
            mixed_label.append(labels[i:i+2])
        inputs,labels = torch.stack(mixed_input).to(device), torch.stack(mixed_label).to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs,k=10)[1]
        top1 = torch.topk(outputs,k=2)[1]
        top5_val.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels[:,0])]))
        top5_val.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels[:,1])]))
        top1_val.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels[:,0])]))
        top1_val.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels[:,1])]))
        val_loss.append(sum(criterion(outputs, labels[:,i]).to(device) for i in range(2)).item())
    return top5_val, top1_val, val_loss

def run(epoch):
    top5_acc, top1_acc = train_mixed(epoch)
    top5_val, top1_val, val_loss = val_mixed(epoch)

    print('VAL:', epoch, mean(top1_val), mean(top5_val))
    acc_f.write(f'{epoch} : {mean(top1_acc)} : {mean(top5_acc)} : {mean(top1_val)} : {mean(top5_val)}\n')
    scheduler.step(mean(val_loss))

    torch.save(model.state_dict(), f'checkpoint/model.{epoch}')
    gc.collect()
    torch.cuda.empty_cache()

for epoch in range(checkpoint+1,checkpoint+num_epochs+2):
    run(epoch)
