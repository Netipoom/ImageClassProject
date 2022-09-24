from PIL import Image
import numpy as np
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        max_size=max(im.size)
        min_size=min(im.size)
        #print(im.size)
        im.thumbnail([256*(max_size/min_size),256*(max_size/min_size)])
        #print(im.size)
        im=im.crop([(im.size[0]-224)/2,(im.size[1]-224)/2,224+(im.size[0]-224)/2,224+(im.size[1]-224)/2])
        #print(im.size)
        #plt.imshow(im)
        np_image = np.array(im)
        #print(np_image)
        #print(np_image.shape)
        mean = 255*np.array([0.485, 0.456, 0.406])
        std = 255*np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        np_image = np.transpose(np_image,(2,0,1))
        #print(np_image)
        return np_image