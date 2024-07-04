import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from dolfin import dx, dot, grad, div, assemble
from helpers import assemble_sparse

# def get_data_image(data, nodes, example_name, final_time):
#     '''
#     Loads the target data: 
#         hat{m} if input is "m", or hat{f} if input is "f"
#     Size of the loaded image should be sqnodes x sqnodes.
#     Output = numpy array sqnodes x sqnodes.
#     example_name = e.g. 'mimura_tsujikawa_t30_'
#     '''
#     img_path = 'data/'
#     pixel_dim = int(np.sqrt(nodes))
#     if pixel_dim**2 != nodes:
#         raise ValueError(f"{nodes} is not a perfect square.")
    
#     if data=='m': # loads \hat{u}
#         c = 0
#         d = 2.18725
#     elif data=='f': # loads \hat{v}
#         c = 0.000608137
#         d = 0.0321876
#     else:
#         return print("Error: Input is incorrect.")        

#     img_name = example_name + data + str(pixel_dim) + '.png'
#     img_rgb = mpimg.imread(img_path + img_name)
    
#     # Create greyscale image by averaging over RGB (-> values in [0,1])
#     img_grey = np.mean(img_rgb, axis=2)
#     # pixel values min, max:
#     a = np.amin(img_grey)
#     b = np.amax(img_grey)

#     # Linear transform from [a,b] to [c,d]
#     img_t = (d-c)/(b-a) * (img_grey-a) + c
#     return img_t

def get_data_array(var, example_name, final_time):
    '''
    Loads the target data: 
        hat{m} if input is "m", or hat{f} if input is "f"
    Size of the loaded array should be sqnodes x sqnodes.
    Output = numpy array sqnodes x sqnodes.
    '''
    data = np.genfromtxt('data/' + example_name + '_t' + str(final_time) + '_' + str(var) + '.csv', delimiter=',')

    return data

def generate_image(data, nodes):
    '''
    Loads the original image for target data at : 
        hat{m} if input is "m", or hat{f} if input is "f".
    Generates the images used as target states that match the mesh size.
    '''
    pixel_dim = int(np.sqrt(nodes))
    if pixel_dim**2 != nodes:
        raise ValueError(f"{nodes} is not a perfect square.")
    
    img_path = 'data/'
    
    img_orig = 'mimura_tsujikawa_t30_' + data + '.png'
    img_new = 'mimura_tsujikawa_t30_' + data + str(pixel_dim) + '.png'

    # Resize image automatically instead of suplying images of different lengths
    img = Image.open(img_path + img_orig)
    img_re = img.resize((pixel_dim, pixel_dim))
    img_re.save(img_path + img_new)
    print(f'Created image of dimension {pixel_dim} for variable {data}')    
    # return img_re