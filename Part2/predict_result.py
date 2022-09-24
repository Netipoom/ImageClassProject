import torch
from process_image import process_image
def predict_result(image, model, topk, gpu,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #print(image , image_folder,cat_to_name[image_folder])
    device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")
    model.to(device)
    #model.eval()
    np_image=torch.from_numpy(image).unsqueeze(0)
    #print(np_image)
    #print(np_image.to(device = device,dtype = torch.float))
    model.eval()
    with torch.no_grad():
        logps = model.forward(np_image.to(device = device,dtype = torch.float))
        # Calculate accuracy
        ps = torch.exp(logps)
        #print(ps)
        top_p, top_class = ps.topk(topk, dim=1)
        #print(top_p, top_class)
        #print(str(top_class[0][0].item()))
        #print(cat_to_name[str(top_class[0][0].item())])
        idx_to_class = {model.class_to_idx[k]:k for k in model.class_to_idx.keys()}
        top_class = [idx_to_class[x] for x in top_class[0].tolist()]
        for i in range(topk):
            print(f'flower name: {cat_to_name[str(top_class[i])]}, confidence: {"%.2f"%(top_p[0][i]*100)}%')
    '''
    with Image.open(image_path+'/'+image_folder+'/'+image) as im:
        top_class = [idx_to_class[x] for x in top_class[0].tolist()]
        view_classify(im,top_class, top_p,cat_to_name[image_folder])'''