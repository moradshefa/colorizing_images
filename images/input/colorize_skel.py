
# coding: utf-8

# In[1]:


# CS194-26 (CS294-26): Project 1
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time
import os
import argparse

# name of the input file
# imname = 'cathedral.jpg'
# imname = 'monastery.jpg'
# imname = 'nativity.jpg'
# imname = 'settlers.jpg'
# imname = 'emir.tif'
# imname = 'harvesters.tif'
# imname = 'icon.tif'
# imname = 'lady.tif'
# imname = 'self_portrait.tif'
# imname = 'three_generations.tif'
# imname = 'train.tif'
# imname = 'turkmen.tif'
# imname = 'village.tif'
# imname = 'sacks.tif'
# imname = 'lugano.tif'
# imname = 'lugano2.tif'

# imname = 'roses.tif'
# imname = 'cotton.tif'




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()



# read in the image
im = skio.imread("images/input/"+imname)
if imname[-3] == 't':
    skio.imsave("images/input/"+imname[:-4]+ '_.jpg', im)

if not os.path.isdir("images/"+imname[:-4]):
    os.mkdir("images/"+imname[:-4] + '/')


# In[7]:


# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

# im = im[::2,::2]
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b_ = im[:height]
g_ = im[height: 2*height]
r_ = im[2*height: 3*height]


# In[8]:


# def cut_identity(r,g,b, dummy=None):
#     return np.dstack([r, g, b])

def displace(im1, x, y):
    # displaces an image by x,y
    res = np.roll(im1, x, axis=0)
    res = np.roll(res, y, axis=1)
    return res    
    
def cut_for_loss(im1,im2, x1,x2,y1,y2):
    assert(im1.shape == im2.shape)
    xborder,yborder = 20,20
    xoffset = max(abs(x1), abs(x2)) + xborder
    yoffset = max(abs(y1), abs(y2)) + yborder
    
    im1 = im1[xoffset:-xoffset,yoffset:-yoffset]
    im2 = im2[xoffset:-xoffset,yoffset:-yoffset]
    return im1,im2

def l2_loss(im1, im2, x1,x2,y1,y2):
    # return l2 loss of the difference between 2 images
    # cut off borders
    im1, im2 = cut_for_loss(im1,im2,x1,x2,y1,y2)
    im1 = im1.flatten()
    im2 = im2.flatten()
    return np.linalg.norm(im1-im2)

# needs to be updated using cut_for_loss
def norm_cross_corr(im1,im2,x1,x2,y1,y2):
    # return negative norm cross correlation
    # negative since a high cross correlation means similar vectors so we use the negative 
    # so disimlar vectors get a large loss
    im1, im2 = cut_for_loss(im1, im2, x1,x2,y1,y2)
    im1 = im1.flatten()
    im2 = im2.flatten()
    return -np.dot(im1,im2)/np.linalg.norm(im1)/np.linalg.norm(im2)

# def align_pyramid(im1, im2, loss, displace, x1,x2,y1,y2):  
# #     print("Pyramid: im1.shape", im1.shape, "looking in window (x1,x2,y1,y2)", x1,x2,y1,y2)  
#     dim1,dim2 = im1.shape 
#     size = min(dim1,dim2)
#     if size > 800:
#         im1_ = im1[::2,::2]
#         im2_ = im2[::2,::2]
#         x,y = align_pyramid(im1_, im2_, loss, displace, x1//2, x2//2,y1//2,y2//2) 
#         x,y = align_exhaustive(im1, im2, loss, displace, 2*x-2, 2*x+2,2*y-2,2*y+2)
#         return x,y
#     x,y = align_exhaustive(im1, im2, loss, displace, x1, x2,y1,y2)
#     return x,y



def align_exhaustive(im1, im2, loss, displace,x1,x2,y1,y2):    
#     print("Exhaustive: im1.shape", im1.shape, "looking in window (x1,x2,y1,y2)", x1,x2,y1,y2)
    loss_ = []
    idxs = []
    
    for i,x in enumerate(np.arange(x1, x2)):
        loss_row = []
        idxs_row = []
        for j,y in enumerate(np.arange(y1, y2)):
            im1_ = displace(im1, x, y)
            idxs_row.append((x,y))
            loss_row.append(loss(im1_, im2, x1, x2,y1,y2))
        loss_.append(loss_row)
        idxs.append(idxs_row)
    loss_ = np.asarray(loss_)
    idx = np.argmin(loss_)
    val_per_row = (y2-y1)
    
    row = idx // val_per_row
    col = idx % val_per_row
        
    x,y = idxs[row][col]
    return x,y
    

def edge_detector2(im, filter_=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])):
    im_filtered = np.empty(im.shape, dtype = im.dtype)
    im_filtered = convolve2d(im , filter_, mode='valid')
    return im_filtered




def align_pyramid_iterative(im1, im2, loss, displace, x1,x2,y1,y2):
#     x,y = align_exhaustive(im1, im2, loss, displace,44,48,-58,-54)
    
    dim1,dim2 = im1.shape 
    size = max(dim1,dim2)
    
    level = 0
    while size > 800:
        level += 1
        size = size // 2
    
    decimation = 2**level
       
    im1_ = im1[::decimation,::decimation]
    im2_ = im2[::decimation,::decimation]
    
    x1_ = x1 / decimation
    x2_ = x2 / decimation
    y1_ = y1 / decimation
    y2_ = y2 / decimation
    

    x1_ = int(max(abs(x1_+1),abs(x1_-1)) * abs(x1_) / x1_)
    x2_ = int(max(abs(x2_+1),abs(x2_-1)) * abs(x2_) / x2_)
    y1_ = int(max(abs(y1_+1),abs(y1_-1)) * abs(y1_) / y1_)
    y2_ = int(max(abs(y2_+1),abs(y2_-1)) * abs(y2_) / y2_)
    
        
    for i in range(level+1):
        dec = 2**(level-i)
        im1_ = im1[::dec,::dec]
        im2_ = im2[::dec,::dec]
        x,y = align_exhaustive(im1_, im2_, loss, displace,x1_,x2_,y1_,y2_)
        x1_,x2_,y1_,y2_ = 2*x-2, 2*x+2, 2*y-2, 2*y+2   
    return x,y
        
def cut_image(r,g,b):
    r_lower, r_upper = cut_channel(r), cut_channel(r[::-1])
    g_lower, g_upper = cut_channel(g), cut_channel(g[::-1])
    b_lower, b_upper = cut_channel(b), cut_channel(b[::-1])
  
    lower = max(r_lower,g_lower,b_lower)
    upper = max(r_upper,g_upper,b_upper)
    
    r,g,b = r[lower:-upper],g[lower:-upper],b[lower:-upper]
    
    r_lower_col, r_upper_col = cut_channel(r.T), cut_channel(r.T[::-1])
    g_lower_col, g_upper_col = cut_channel(g.T), cut_channel(g.T[::-1])
    b_lower_col, b_upper_col = cut_channel(b.T), cut_channel(b.T[::-1])

    lower = max(r_lower_col,g_lower_col,b_lower_col)
    upper = max(r_upper_col,g_upper_col,b_upper_col)

#     print("final cuts cols: ",lower, upper, r.shape)

    r,g,b = r[:,lower:-upper],g[:,lower:-upper],b[:,lower:-upper]
    return r,g,b
    
def cut_channel(r):
    x,y = r.shape
    xmax_cutoff = x * 10//100

    sub = np.copy(r[:xmax_cutoff])

    while y > 2000:
        sub = sub[:,::2]
        y = y // 2
    
    rows = 2
    cols = y // 40
    
    top = np.ones((rows, cols))
    filter_ = np.vstack((-top, top))
    
    sub = edge_detector2(sub,filter_)  

    sub[sub > 0.9] = 1
    
    means_lower = np.apply_along_axis(np.mean, 1, sub[:xmax_cutoff])
    lower = np.argmax(np.abs(means_lower))
    return lower

def align(r,g,b, window = None, verbose=False):
    loss = norm_cross_corr
    if not window:
        x,y = r.shape
        window = (-1*x//22,1*y//22)

    x1,x2,y1,y2 = 2*window
    
    if verbose:
        im = np.dstack([r, g, b])
        fig=plt.figure(figsize=(20,20))
        fig.add_subplot(1,2,1)
        plt.title("Before Aligning")
        plt.imshow(im)
        
        skio.imsave("images/"+imname[:-4] + '/' + imname[:-4]+'_naive.jpg', im)
    
    r_e = edge_detector(r)
    g_e = edge_detector(g)
    b_e = edge_detector(b)
    
    x_1,y_1 = align_pyramid_iterative(r_e, b_e, loss, displace, x1,x2,y1,y2)
    x_2,y_2 = align_pyramid_iterative(g_e, b_e, loss, displace, x1,x2,y1,y2)
    
    ar = displace(r, x_1,y_1)
    ag = displace(g, x_2,y_2)
    
    if verbose:
        im = np.dstack([ar, ag, b])
        fig.add_subplot(1,2,2)
        skio.imsave("images/"+imname[:-4] + '/' + imname[:-4]+'_aligned.jpg', im)

        plt.title("After aligning")
        plt.imshow(im)
        
        fig=plt.figure(figsize=(20,20))
        fig.add_subplot(1,2,1)
        plt.title("Before cutting")
        plt.imshow(im)


    r,g,b = cut_image(ar,ag,b)
    if verbose:
        im = np.dstack([r, g, b])
        fig.add_subplot(1,2,2)
        plt.title("After cutting")
        plt.imshow(im)    
        skio.imsave("images/"+imname[:-4] + '/' + imname[:-4]+'_cut.jpg', im)

    
    if verbose:
        im = np.dstack([r, g, b])
        fig=plt.figure(figsize=(20,20))
        fig.add_subplot(1,2,1)
        plt.title("Before contrasting")
        plt.imshow(im)
    
    r = contrast(r)
    g = contrast(g)
    b  = contrast(b)
    im = np.dstack([r, g, b])
    
    if verbose:
        fig.add_subplot(1,2,2)
        plt.title("After contrasting")
        plt.imshow(im)
        skio.imsave("images/"+imname[:-4] + '/' + imname[:-4]+'_contrasted.jpg', im)


    return im,x_1,y_1,x_2,y_2
    
def naive_align(r,g,b):
    return np.dstack([r, g, b]) 

def align_given_displace(r,g,b,x1,y1,x2,y2):
    return naive_align(displace(r,x1,y1),displace(g,x2,y2),b)

def contrast(x):
    x = x-np.min(x)
    y = np.sort(x.flatten())
    lower_p = 2
    percentile = 100-lower_p

    lower_p = y[y.shape[0]*lower_p//100]
    percentile = y[y.shape[0]*percentile//100]
    above = x > percentile
    x[above] = percentile 
    
    below = x < lower_p
    x[below] = lower_p 
    x = x-np.min(x)
    return x / np.max(x)

def edge_detector(im):
    g1 = [2,4,5,4,2]
    g2 = [4,9,12,9,4]
    g3 = [5,12,15,12,5]
    
    gaussian = np.vstack((g1,g2,g3,g2,g1))/159
    im = convolve2d(im , gaussian, mode='valid')
    
    sobel_x = np.asarray([[1,0,-1], [2,0,-2], [1,0,-1]])
    sobel_y = sobel_x.T
    
    grad_x = convolve2d(im , sobel_x, mode='valid')
    grad_y = convolve2d(im , sobel_y, mode='valid')
    
    grad_mag = np.sqrt(grad_x**2+grad_y**2)

    return grad_mag


# In[9]:


# align the images
verbose = True
r,g,b = np.copy(r_),np.copy(g_),np.copy(b_)
start = time.time()
im_out,x1,y1,x2,y2 = align(r,g,b, verbose=verbose)
end = time.time()

print('displaced r by',x1,y1)
print('displaced g by',x2,y2)

print('time: ', end-start)
    
# # save the image
skio.imsave("images/"+imname[:-4] + '/' + imname[:-4]+'_out.jpg', im_out)

if not verbose:
    # display the image
    plt.figure(figsize=(11,11))
    plt.imshow(im_out)
    plt.title("Aligned and cut")
    plt.show()

