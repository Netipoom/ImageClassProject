from get_arg_pred import get_inputs_args
from process_image import process_image
from predict_result import predict_result
from get_model import get_model
from cat2name import cat2name
def main():
    in_arg = get_inputs_args()
    cat_to_name = cat2name(in_arg.category_names)
    process_img = process_image(in_arg.input[0]) 
    model = get_model(in_arg.input[1])
    predict_result(process_img, model, int(in_arg.top_k), in_arg.gpu,cat_to_name)
if __name__ == "__main__":
    main()