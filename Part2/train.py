from get_arg import get_inputs_args
from loadData import loadData
from cat2name import cat2name
from train_model import train_model
from prepareModel import prepareModel
from save_model import save_model
def main():
    in_arg = get_inputs_args()
    model,criterion,optimizer = prepareModel(in_arg.arch,int(in_arg.hidden_units))
    trained_model,optimizer = train_model(model,criterion,optimizer,in_arg.epochs,in_arg.data_dir[0],in_arg.gpu)
    save_model(trained_model,in_arg.save_dir,in_arg.epochs,optimizer,in_arg.data_dir[0])

if __name__ == "__main__":
    main()