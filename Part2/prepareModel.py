import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
def prepareModel(arch,hidden_unit):
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_unit)),
                            ('relu1', nn.ReLU()),
                            ('dropout1',nn.Dropout(p=0.2)),
                            ('fc2', nn.Linear(hidden_unit, 512)),
                            ('relu2', nn.ReLU()),
                            ('dropout2',nn.Dropout(p=0.4)),
                            ('fc3', nn.Linear(512, 255)),
                            ('relu3', nn.ReLU()),
                            ('dropout3',nn.Dropout(p=0.3)),
                            ('fc4', nn.Linear(255, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, hidden_unit)),
                            ('relu1', nn.ReLU()),
                            ('dropout1',nn.Dropout(p=0.2)),
                            ('fc2', nn.Linear(hidden_unit, 512)),
                            ('relu2', nn.ReLU()),
                            ('dropout2',nn.Dropout(p=0.4)),
                            ('fc3', nn.Linear(512, 255)),
                            ('relu3', nn.ReLU()),
                            ('dropout3',nn.Dropout(p=0.3)),
                            ('fc4', nn.Linear(255, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    

    return model,criterion,optimizer