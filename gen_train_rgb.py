

import glob
import cv2


import util

import h5py
import numpy as np
import sys

import torch


from skimage.transform import resize

def flip_img(img,flip):

    if flip == 0:
        pass
    elif flip == 1:
        img = cv2.flip(img, 0)
    elif flip == 2:
        img = cv2.flip(img, 1)
    else:
        img = cv2.flip(img, -1)
    return img


def rotate_img(img,rotate):
    if rotate == 0:
        pass
    elif rotate == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img





###################################################         generate augmented image and save on the hard disk. Convert to Quad and Bayer online       ################################################################



def gen_augumented_img(filename,saveToFolder,size_label=128,stride = 64):


    #################################################                       parameters for generating training images
    BGR =True


    # how much space from the four boundaries of the image
    margin = 0

    # modulo = 4
    downszRatio=[1]

    #################################################                       generating training images <quad, bayer> pattern pair

    #img = cv2.imread(filename)
    img = np.load(filename, allow_pickle = True)

    extIdx =  filename.rfind('.')
    folderIdx =  filename.rfind('/')

    filename = filename[folderIdx+1:extIdx]

    # img = modCrop(img, modulo=modulo)

    orig = img.copy()

    count = 0

    for j in range(len(downszRatio)):

        for flip in range(1):  # no flip; flip x-axis, flip y-axis, flip both x- and y- axis

            for rotate in range(4):  # no rotation; rotate 90, -90, 180

                img = cv2.resize(orig, dsize=(0, 0), fx=downszRatio[j], fy=downszRatio[j], interpolation=cv2.INTER_CUBIC)

                img = flip_img(img, flip)  ###############         flip

                img = rotate_img(img, rotate)  ###############         rotate

                h = img.shape[0]
                w = img.shape[1]

                for y in range(margin, h - margin - size_label, stride):  ###############         generate sub images
                    for x in range(margin, w - margin - size_label, stride):

                        # get a sub image
                        subimg = img[y: y + size_label, x: x + size_label, :]
                        assert subimg.shape[0]==128 and subimg.shape[1]==128, "subImg dimension incorrect: "+ str(subimg.shape[0])+ " "+ str(subimg.shape[1])
                        newName=saveToFolder+'/'+filename+"_"+str(count)+".png"
                        #cv2.imwrite(newName,subimg)
                        np.save(newName, subimg, allow_pickle=True)

                        count += 1

    return count


def main_augumented_img(file_path,aug_img_folder):

    #image_paths = util.get_all_files(file_path)
    image_paths = util.get_all_rgby_files(file_path)
            
    count =0
    total_img=len(image_paths)

    for i in range(total_img):
        print('processed images ratio: ' + str( int(i*100/total_img) )+"%" )
        c = gen_augumented_img(image_paths[i], aug_img_folder,size_label=128, stride=64)
        count += c

    print('total num of images: ' + str(i))
    print('total num of <quad, bayer/rgb> pairs after augumentation: ' + str(count))


    pass




###############################################################################################################################################################################





debug=False


if __name__ == "__main__":

    ################################################################################            take user input argument
    if debug:
        # TRAINING='training'
        # sz_label = 128

        aug_img_folder = '/newDisk/users/junjiang/test/sr_imx586_LapSRN/aug_img'
        util.create_dir(aug_img_folder, delete_exising_files=True)
        src_folder = ['/newDisk/dataset/LapSRN/SR_training_datasets/BSDS200',
                       '/newDisk/dataset/LapSRN/SR_training_datasets/T91']

    else:

        if len(sys.argv) <2:
            print("---------------------------------------------------------------------------------------")
            print("")
            print("To create training dataset, execute")
            print("python gen_train_rgb <'training'> <sz_label>")
            print("e.g. python gen_train_rgb.py training 128")
            print("")
            print("---------------------------------------------------------------------------------------")

            sys.exit(1)
        else:
            src_folder=sys.argv[1]
            aug_img_folder =sys.argv[2]


################################################################################



    # folder= '/newDisk/test/sr_imx586_LapSRN'
    # util.create_dir(folder,delete_exising_files=False)
    # gen_offline_hdf5(TRAINING)


    util.create_dir(aug_img_folder,delete_exising_files=True)

    main_augumented_img(src_folder,aug_img_folder)

