import os

import glob
import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt


# return all images under the folder (but not under the subfolders)
def make_dataset(img_path):
    filepaths = glob.glob(img_path + '/*.*')

    return filepaths


# get all png or jpg image names under the folder including the subfolders
# [in]
#   folder:
# [out]
#   a list of image path of all images (png or jpg) under the folder
#
def get_all_files(folder,reg=None):
    all_images = []
    for root, dirs, files in os.walk(folder):
        if len(files) > 0:
            image_paths = []
            for name in files:

                if reg is None:
                    if name.find('.png') >= 0 or name.find('.jpg')>=0  or name.find('.tif') >= 0:
                        if len(image_paths) == 0:
                            image_paths.append(os.path.join(root, name))
                        else:
                            image_paths.append(os.path.join(root, name))
                else:
                    if name.find(reg) >= 0:
                        if len(image_paths) == 0:
                            image_paths.append(os.path.join(root, name))
                        else:
                            image_paths.append(os.path.join(root, name))




            if len(all_images) == 0:
                all_images = image_paths
            else:
                all_images = all_images + image_paths

    # for i in range(10):
    #     print(all_images[i])
    #
    # print(len(all_images))

    return all_images




# get all rgby *.npy image names under the folder including the subfolders
# [in]
#   folder:
# [out]
#   a list of image path of all rgby images (npy) under the folder
#
def get_all_rgby_files(folder,reg=None):
    all_images = []

    for root, dirs, files in os.walk(folder):
        if len(files) > 0:
            image_paths = []
            for name in files:

                if reg is None:
                    if name.find('.npy') >= 0:
                        if len(image_paths) == 0:
                            image_paths.append(os.path.join(root, name))
                        else:
                            image_paths.append(os.path.join(root, name))
                else:
                    if name.find(reg) >= 0:
                        if len(image_paths) == 0:
                            image_paths.append(os.path.join(root, name))
                        else:
                            image_paths.append(os.path.join(root, name))


            if len(all_images) == 0:
                all_images = image_paths
            else:
                all_images = all_images + image_paths

    return all_images







def create_dir(directory,delete_exising_files=True):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if delete_exising_files:
            files = glob.glob(directory+'/*.*')
            for f in files:
                os.remove(f)

            #remove all dir
            files = glob.glob(directory + '/*')
            for f in files:
                shutil.rmtree(f)




###########################################3            write csv file for training             ########################################

def write_csv(imgFolder,filename):

    # filename='training.csv'

    imgFiles = glob.glob(imgFolder+'/*quad*')


    with open(filename, "w") as text_file:
        text_file.write("quad file,bayer file\n")

        for i in range(len(imgFiles)):
            quad_file=imgFiles[i]
            bayer_file=quad_file.replace('quad','bayer')
            text_file.write("{},{}\n".format(quad_file,bayer_file) )
            # text_file.write("{},{}\n".format(i,i*2) )





def rgb2lab(img,linearSpace):

    if linearSpace:
        pass
    else:
        img = np.power(img,1/2.2)



    imgLab= cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    imgLab[:, :, 0] = imgLab[:, :, 0]  / 100
    imgLab[:, :, 1] = (imgLab[:, :, 1] + 128 ) / 255
    imgLab[:, :, 2] = (imgLab[:, :, 2] + 128 ) / 255

    # print(np.max(imgLab))
    # print(np.min(imgLab))

    return imgLab


def lab2rgb(imgLab):


    imgLab[:, :, 0] = imgLab[:, :, 0] * 100
    imgLab[:, :, 1] = imgLab[:, :, 1] * 255-128
    imgLab[:, :, 2] = imgLab[:, :, 2] * 255-128

    RGB = cv2.cvtColor(imgLab, cv2.COLOR_LAB2RGB)

    # print(np.max(RGB))
    # print(np.min(RGB))

    return RGB








def show_grid_image(img,rows=None,cols=None,title=None):

    # img= cv2.imread('t_1.png',-1)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.title(title)
    plt.show()


def get_filename(file):

    idx = file.rfind('/')

    filename = file[idx+1:]

    filepath = file[0:idx]


    return filename, filepath
