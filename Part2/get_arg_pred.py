import argparse
def get_inputs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",nargs='+')
    parser.add_argument("--top_k",default=5)
    parser.add_argument("--category_names",default="cat_to_name.json")
    parser.add_argument("--gpu",action='store_true')
    return parser.parse_args()