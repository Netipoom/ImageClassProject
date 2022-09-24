import argparse
def get_inputs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",nargs='+')
    parser.add_argument("--save_dir",default='checkpoint.pth')
    parser.add_argument("--arch",default="vgg11")
    parser.add_argument("--learning_rate",default=0.003)
    parser.add_argument("--hidden_units",default=512)
    parser.add_argument("--epochs",default=10)
    parser.add_argument("--gpu",action='store_true')
    return parser.parse_args()