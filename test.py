# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午6:41

import cv2
import numpy as np
import os
from PIL import Image
import os.path

rootpath='/home/user_home/chengyingjie/project2'

def tif_to_png(image_path,save_path):
    """
    :param image_path: *.tif image path
    :param save_path: *.png image path
    :return:
    """
    img = cv2.imread(image_path,3)
    # print(img)
    # print(img.dtype)
    filename = image_path.split('/')[-1].split('.')[0]
    # print(filename)
    save_path = save_path + '/' + filename + '.png'
    cv2.imwrite(save_path,img)
def jpg_to_png(source_dir,output_path):
    """
    :param image_path: *.tif image path
    :param save_path: *.png image path
    :return:
    """
    i=1
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):

            img = Image.open(os.path.join(source_dir, filename))
            destination_filename = os.path.splitext(filename)[0] + '.jpg'
            print(i)
            print(destination_filename)
            i=i+1
            img.save(os.path.join(output_path, destination_filename))
def ren(path):
    """
    :param image_path: *.tif image path
    :param save_path: *.png image path
    :return:
    """

    for filename in os.listdir(path):  # ‘logo/’是文件夹路径，你也可以替换其他
        newname = filename.replace('_mask.png', '.png')  # 把jpg替换成png
        os.rename(path + filename, path + newname)


def gif_to_png(rootdir,save_path):
    # rootdir = r'D:\用户目录\我的图片\From Yun\背景图\背景图'  # 指明被遍历的文件夹
    #rootdir = r'E:\AD datasets\voiceClassifyGoogle\Class\C'  # 原图片目录

    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            print('parent is :' + parent)
            print('filename is :' + filename)
            currentPath = os.path.join(parent, filename)
            print('the fulll name of the file is :' + currentPath)

            im = Image.open(currentPath)  # 打开gif格式的图片

            def iter_frames(im):
                try:
                    i = 0
                    while 1:
                        im.seek(i)
                        imframe = im.copy()
                        if i == 0:
                            palette = imframe.getpalette()
                        else:
                            imframe.putpalette(palette)
                        yield imframe
                        i += 1
                except EOFError:
                    pass

            for i, frame in enumerate(iter_frames(im)):
                # frame.save(save_path + filename[:-4] + '.png', **frame.info)
                frame.save(save_path + filename[:-4].replace('manual1','test') + '.png', **frame.info)

if __name__ == '__main__':
    # root_path = r'视网膜血管数据集/training/images/'
    # save_path = r'Dataset/train/image/'
    root_path = rootpath+'/datasets/iChallenge/train/masks/'
    save_path = rootpath+'/datasets/iChallenge/train/masks_jpg/'
    image_files = os.listdir(root_path)
    # for image_file in image_files:
        #tif_to_png(root_path + image_file,save_path)
    # gif_to_png(root_path, save_path)
    #jpg_to_png(root_path,save_path)
    ren(rootpath+'/datasets/Dataset_BUSI_with_GT/benign/masks/')


