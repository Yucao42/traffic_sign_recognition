from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import resnet
import datetime as dt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--net', type=str, default='resnet50', metavar='NET',
                    help="Name of the network module")
parser.add_argument('--name', type=str, default='smallbatch', metavar='N',
                    help="Name of the module")
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dp', type=float, default=0.5, metavar='DP',
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--step', type=int, default=100, metavar='STEP',
                    help='steo (default: 100)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size//2, shuffle=False, num_workers=4)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
#model = Net(args.dp)
if 'resnet50' in args.net:
    model = resnet.resnet50(False, dp = args.dp)
if 'resnet18' in args.net:
    model = resnet.resnet18(False, dp = args.dp)

if args.load:
    model.load_state_dict(torch.load(args.load))
#model.load_state_dict(torch.load(['model_latest.pth'])['state_dict'])

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step)
device = torch.device('cuda:0')
model.to(device)
wrongs = {}
totals = {}
for i in range(43):
    wrongs[i] = 0
    totals[i] = 0

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for idx, (data, target) in enumerate( val_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        res = pred.eq(target.data.view_as(pred)).cpu()
        for i in range(len(res)):
            if res[i] == 0:
                wrongs[int(target[i])] = wrongs[int(target[i])] + 1
            totals[int(target[i])] = totals[int(target[i])] + 1


    validation_loss /= len(val_loader.dataset)
    print( wrongs )
    embed()
    print(dt.datetime.now(), '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


validation()
