import glob
import shutil
import os
# for data in glob.glob("/home/user_home/chengyingjie/project2/datasets/Dataset_BUSI_with_GT/benign/"):  # 原始地址
#      if not os.path.isdir(data):
#          shutil.move(data,"C:\\2\\")   # 目标地址
r_path='/home/user_home/chengyingjie/project2/datasets/iChallenge/'
dir_path=r_path+'train/masks/'
new_image_folder=r_path+'all/masks/'
for root, dirs, files in os.walk(dir_path):
     # root 表示当前正在访问的文件夹路径
     # dirs 表示该文件夹下的子目录名list
     # files 表示该文件夹下的文件list
     for file in files:
         if file.endswith('.png'):
             shutil.move(os.path.join(root, file), new_image_folder)  # 移动图片到新文件夹

