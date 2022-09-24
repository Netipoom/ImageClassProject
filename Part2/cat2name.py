import json
def cat2name(cat_to_name):
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name