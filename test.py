import torch
import os
from torchvision import datasets, transforms
from models.senet import *
from statistics import mean


batch_size = 100
test_root = 'test/'

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

test_data = datasets.ImageFolder(root='test/test', transform=base_transform)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

label_index = {}
for i,dir in enumerate(sorted(os.listdir('val/'))):
    # print (i,dir)
    label_index[i] = dir.split('-')[1]

# print(label_index)

model = se_resnet18(120)
model.load_state_dict(torch.load('checkpoint/model.21', map_location='cpu'))

f_list = []
for f in sorted(os.listdir('test/test/0')):
    f_list.append(f)
print (f_list)

for batch_num, (inputs, labels) in enumerate(test_loader):
    model.eval()
    outputs = torch.nn.functional.softmax(model(inputs),dim=1)
    # print (outputs)
    top = torch.topk(outputs,k=5)
    for i,dog in enumerate(top[1].numpy()):
        print (f_list[i], [label_index[d] for d in dog], top[0][i].detach().numpy())


# top5_val, top1_val, val_loss = [],[],[]
# model.eval()
# for batch_num, (inputs, labels) in enumerate(test_loader):
#     outputs = model(inputs)
#     top5 = torch.topk(outputs,k=5)[1]
#     top1 = torch.topk(outputs,k=1)[1]
#     top5_val.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
#     top1_val.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))
#     # val_loss.append(criterion(outputs, labels).to(device).item())
#     print(batch_num, mean(top5_val), mean(top1_val))
