import torch
def get_model(checkpoint):
    #print(torch.load(checkpoint+'.pth'))
    model = torch.load(checkpoint+'.pth')['model']
    model.class_to_idx = torch.load(checkpoint+'.pth')['class_to_idx']
    return model