import torch
from loadData import loadData
def save_model(model,save_dir,epoch,optimizer,data_dir):
    state = {
    'model':model,
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'class_to_idx':loadData(data_dir)['class_to_idx']
    }
    torch.save(state, save_dir)