import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from sklearn.model_selection import KFold

root_path='/home/user_home/chengyingjie/project2'

# class LiverDataset(data.Dataset):
#     def __init__(self, state, transform=None, target_transform=None):
#         self.state = state
#         # self.train_root = "/datasets/liverme/train/"
#         # self.val_root = "/datasets/liverme/val/"
#         self.train_root = root_path+"/datasets/liverme/train/"
#         self.val_root = root_path+"/datasets/liverme/val/"
#         self.test_root = self.val_root
#         self.pics,self.masks = self.getDataPath()
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def getDataPath(self):
#         assert self.state =='train' or self.state == 'val' or self.state =='test'
#         if self.state == 'train':
#             root = self.train_root
#         if self.state == 'val':
#             root = self.val_root
#         if self.state == 'test':
#             root = self.test_root
#
#         pics = []
#         masks = []
#
#         # 1、自制数据集的数据读取方式
#         n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
#         for i in range(n):
#             img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
#             mask = os.path.join(root, "y_%d_mask.png" % (i+1))
#             if not os.path.exists(img):
#                 continue
#             pics.append(img)
#             masks.append(mask)
#
#         #2、liver数据集的读取方式   数据集样式不同做出相应的改变
#         # for pic in glob(root + 'images'+ '/*.png'):
#         #     pics.append(pic)
#         #     file =os.path.join(root,'masks',os.path.basename(pic).split('.png')[0]+'.png')
#         #     masks.append(file)
#
#         return pics,masks
#
#     def __getitem__(self, index):
#         #x_path, y_path = self.imgs[index]
#         x_path = self.pics[index]
#         y_path = self.masks[index]
#         origin_x = Image.open(x_path).resize((224,224))
#         origin_y = Image.open(y_path).convert('L').resize((224,224))
#
#
#         # origin_x = cv2.imread(x_path).resize((224,224))
#         # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY).resize((224,224))
#         if self.transform is not None:
#             img_x = self.transform(origin_x)
#         if self.target_transform is not None:
#             img_y = self.target_transform(origin_y)
#
#         return img_x, img_y,x_path,y_path
#
#     def __len__(self):
#         return len(self.pics)

class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = "/datasets/iChallenge/train/"  #将数据集所有数据都放在这个目录下面
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_num = 0

    def getDataPath(self):
        root = self.train_root

        pics = []
        masks = []

        # 1、自制数据集的数据读取方式
        # n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        # for i in range(n):
        #     img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
        #     mask = os.path.join(root, "y_%d_mask.png" % (i+1))
        #     if not os.path.exists(img):
        #         continue
        #     pics.append(img)
        #     masks.append(mask)

        #2、liver数据集的读取方式   数据集样式不同做出相应的改变
        for pic in glob(root + 'images'+ '/*.jpg'):
            pics.append(pic)
            file =os.path.join(root,'masks',os.path.basename(pic).split('.jpg')[0]+'.png')
            masks.append(file)

        floder = KFold(n_splits=5, random_state=42, shuffle=True)
        train_files = []  # 存放5折的训练集划分
        train_masks = []  # 存放5折的训练集标签划分
        test_files = []  # # 存放5折的测试集集划分
        test_masks=  []  # # 存放5折的测试集集标签划分
        all_files = pics
        mask_files = masks
        for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
            train_files.append(np.array(all_files)[Trindex].tolist())
            train_masks.append(np.array(mask_files)[Trindex].tolist())
            test_files.append(np.array(all_files)[Tsindex].tolist())
            test_masks.append(np.array(mask_files)[Tsindex].tolist())

        if self.state == 'train':
            self.pics = train_files
            self.masks = train_masks

        if self.state == 'val':
            self.pics = test_files
            self.masks = test_masks
        return self.pics,self.masks


    def __getitem__(self, index):
        i = self.dataset_num
        x_path = self.pics[i][index]
        y_path = self.masks[i][index]
        origin_x = Image.open(x_path).resize((224,224))
        origin_y = Image.open(y_path).convert('L').resize((224,224))
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)

        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics[self.dataset_num])


class ichallengeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        # self.train_root = "/datasets/liverme/train/"
        # self.val_root = "/datasets/liverme/val/"
        self.train_root = root_path+"/datasets/iChallenge/train/"
        self.val_root = root_path+"/datasets/iChallenge/val/"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []

        # # 1、自制数据集的数据读取方式
        # n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        # for i in range(n):
        #     img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
        #     mask = os.path.join(root, "y_%d_mask.png" % (i+1))
        #     if not os.path.exists(img):
        #         continue
        #     pics.append(img)
        #     masks.append(mask)

        # 2、liver数据集的读取方式   数据集样式不同做出相应的改变
        for pic in glob(root + 'images'+ '/*.jpg'):
            pics.append(pic)
            file =os.path.join(root,'masks_jpg',os.path.basename(pic).split('.jpg')[0]+'.jpg')
            masks.append(file)

        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path).resize((224,224))
        origin_y = Image.open(y_path).convert('L').resize((224,224))


        # origin_x = cv2.imread(x_path).resize((224,224))
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY).resize((224,224))
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)

        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

# 改为五折交叉验证


# class ichallengeDataset(data.Dataset):
#     # 改为五折交叉验证
#     def __init__(self, state, transform=None, target_transform=None):
#         self.state = state
#         self.train_root = root_path+"/datasets/iChallenge/all/"  #将数据集所有数据都放在这个目录下面
#         self.pics,self.masks = self.getDataPath()
#         self.transform = transform
#         self.target_transform = target_transform
#         self.dataset_num = 0
#
#     def getDataPath(self):
#         root = self.train_root
#
#         pics = []
#         masks = []
#
#         # 1、自制数据集的数据读取方式
#         # n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
#         # for i in range(n):
#         #     img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
#         #     mask = os.path.join(root, "y_%d_mask.png" % (i+1))
#         #     if not os.path.exists(img):
#         #         continue
#         #     pics.append(img)
#         #     masks.append(mask)
#
#         #2、liver数据集的读取方式   数据集样式不同做出相应的改变
#         for pic in glob(root + 'images'+ '/*.jpg'):
#             pics.append(pic)
#             file =os.path.join(root,'masks',os.path.basename(pic).split('.jpg')[0]+'.png')
#             masks.append(file)
#
#         floder = KFold(n_splits=5, random_state=42, shuffle=True)
#         train_files = []  # 存放5折的训练集划分
#         train_masks = []  # 存放5折的训练集标签划分
#         test_files = []  # # 存放5折的测试集集划分
#         test_masks=  []  # # 存放5折的测试集集标签划分
#         all_files = pics
#         mask_files = masks
#         for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
#             train_files.append(np.array(all_files)[Trindex].tolist())
#             train_masks.append(np.array(mask_files)[Trindex].tolist())
#             test_files.append(np.array(all_files)[Tsindex].tolist())
#             test_masks.append(np.array(mask_files)[Tsindex].tolist())
#
#         if self.state == 'train':
#             self.pics = train_files
#             self.masks = train_masks
#
#         if self.state == 'val':
#             self.pics = test_files
#             self.masks = test_masks
#         return self.pics,self.masks
#
#
#     def __getitem__(self, index):
#         i = self.dataset_num
#         x_path = self.pics[i][index]
#         y_path = self.masks[i][index]
#         origin_x = Image.open(x_path).resize((224,224))
#         origin_y = Image.open(y_path).convert('L').resize((224,224))
#         if self.transform is not None:
#             img_x = self.transform(origin_x)
#         if self.target_transform is not None:
#             img_y = self.target_transform(origin_y)
#
#         return img_x, img_y,x_path,y_path
#
#     def __len__(self):
#         return len(self.pics[self.dataset_num])


class cvc300Dataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        # self.train_root = "/datasets/liverme/train/"
        # self.val_root = "/datasets/liverme/val/"
        self.train_root = root_path+"/datasets/CVC-300/train/"
        self.val_root = root_path+"/datasets/CVC-300/val/"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []

        # # 1、自制数据集的数据读取方式
        # n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        # for i in range(n):
        #     img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
        #     mask = os.path.join(root, "y_%d_mask.png" % (i+1))
        #     if not os.path.exists(img):
        #         continue
        #     pics.append(img)
        #     masks.append(mask)

        # 2、liver数据集的读取方式   数据集样式不同做出相应的改变
        for pic in glob(root + 'images'+ '/*.png'):
            pics.append(pic)
            file =os.path.join(root,'masks',os.path.basename(pic).split('.png')[0]+'.png')
            masks.append(file)

        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path).resize((224,224))
        origin_y = Image.open(y_path).convert('L').resize((224,224))


        # origin_x = cv2.imread(x_path).resize((224,224))
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY).resize((224,224))
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)

        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class esophagusDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        # self.train_root = r"E:\datasets\data_sta_all\train_data"
        # self.val_root = r"E:\datasets\data_sta_all\test_data"
        self.train_root = root_path+"/datasets/data_sta_all/train_data"
        self.val_root = root_path+"/datasets/data_sta_all/test_data"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%05d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%05d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        origin_x = Image.open(x_path).convert('RGB').resize((224,224))
        origin_y = Image.open(y_path).convert('L').resize((224,224))


        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class dsb2018CellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\codes\pytorch-nested-unet-master\pytorch-nested-unet-master\input\dsb2018_256'
        self.root = root_path+"/datasets/dsb2018_256"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.img_paths = glob(self.root + '\images\*')
        # self.mask_paths = glob(self.root + '\masks\*')
        self.img_paths = glob(self.root + '/images/*')
        self.mask_paths = glob(self.root + '/masks/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths,self.val_mask_paths  #因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

# class dsb2018CellDataset(data.Dataset):
#     def __init__(self, state, transform=None, target_transform=None):
#         self.state = state
#         self.train_root = root_path+"/datasets/dsb2018_256/"  #将数据集所有数据都放在这个目录下面
#         self.pics,self.masks = self.getDataPath()
#         self.transform = transform
#         self.target_transform = target_transform
#         self.dataset_num = 0
#
#     def getDataPath(self):
#         root = self.train_root
#
#         pics = []
#         masks = []
#
#         # 1、自制数据集的数据读取方式
#         # n = len(os.listdir(root)) // 2 + 700  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
#         # for i in range(n):
#         #     img = os.path.join(root, "y_%d.png" % (i+1))  # liver is %03d
#         #     mask = os.path.join(root, "y_%d_mask.png" % (i+1))
#         #     if not os.path.exists(img):
#         #         continue
#         #     pics.append(img)
#         #     masks.append(mask)
#
#         #2、liver数据集的读取方式   数据集样式不同做出相应的改变
#         for pic in glob(root + 'images'+ '/*.png'):
#             pics.append(pic)
#             file =os.path.join(root,'masks',os.path.basename(pic).split('.png')[0]+'.png')
#             masks.append(file)
#
#         floder = KFold(n_splits=5, random_state=42, shuffle=True)
#         train_files = []  # 存放5折的训练集划分
#         train_masks = []  # 存放5折的训练集标签划分
#         test_files = []  # # 存放5折的测试集集划分
#         test_masks=  []  # # 存放5折的测试集集标签划分
#         all_files = pics
#         mask_files = masks
#         for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
#             train_files.append(np.array(all_files)[Trindex].tolist())
#             train_masks.append(np.array(mask_files)[Trindex].tolist())
#             test_files.append(np.array(all_files)[Tsindex].tolist())
#             test_masks.append(np.array(mask_files)[Tsindex].tolist())
#
#         if self.state == 'train':
#             self.pics = train_files
#             self.masks = train_masks
#
#         if self.state == 'val':
#             self.pics = test_files
#             self.masks = test_masks
#         return self.pics,self.masks
#
#
#     def __getitem__(self, index):
#         i = self.dataset_num
#         x_path = self.pics[i][index]
#         y_path = self.masks[i][index]
#         origin_x = Image.open(x_path).resize((224,224))
#         origin_y = Image.open(y_path).convert('L').resize((224,224))
#         if self.transform is not None:
#             img_x = self.transform(origin_x)
#         if self.target_transform is not None:
#             img_y = self.target_transform(origin_y)
#
#         return img_x, img_y,x_path,y_path
#
#     def __len__(self):
#         return len(self.pics[self.dataset_num])


class CornealDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\datasets\CORN\CORN\Corneal nerve curivilinear segmentation\Corneal nerve curivilinear segmentation'
        self.root = root_path+"/datasets/CORN/Corneal nerve curivilinear segmentation/pngpic"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.train_img_paths = glob(self.root + r'\training\train_images\*')
        # self.train_mask_paths = glob(self.root + r'\training\train_mask\*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths = glob(self.root + "/training/train_images/*")
        self.train_mask_paths = glob(self.root + "/training/train_mask/*")
        self.val_img_paths = glob(self.root + "/val/val_images/*")
        self.val_mask_paths = glob(self.root + "/val/val_mask/*")
        self.test_img_paths = glob(self.root + "/test/test_images/*")
        self.test_mask_paths = glob(self.root + "/test/test_mask/*")
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        # pic = Image.open(pic_path).resize((224,224))
        # mask = Image.open(mask_path).convert('L').resize((224,224))
        pic = Image.open(pic_path).convert('RGB').resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\datasets\DRIVE\DRIVE'
        self.root = root_path+"/datasets/DRIVE/pngpic"
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.train_img_paths = glob(self.root + r'\training\images\*')
        # self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        # self.val_img_paths = glob(self.root + r'\test\images\*')
        # self.val_mask_paths = glob(self.root + r'\test\1st_manual\*')
        self.train_img_paths = glob(self.root + "/training/images/*")
        self.train_mask_paths = glob(self.root + "/training/1st_manual/*")
        self.val_img_paths = glob(self.root + "/test/images/*")
        self.val_mask_paths = glob(self.root + "/test/1st_manual/*")
        self.test_img_paths = self.val_img_paths
        self.test_mask_paths = self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        # pic = cv2.resize(pic,(imgx,imgy))
        # mask = cv2.resize(mask, (imgx, imgy))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\datasets\isbi'
        self.root = root_path+"/datasets/isbi"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.img_paths = glob(self.root + r'\train\images\*')
        # self.mask_paths = glob(self.root + r'\train\label\*')
        self.img_paths = glob(self.root + "/train/images/*")
        self.mask_paths = glob(self.root + "/train/label/*")
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).convert('RGB').resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)


class LungKaggleDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\datasets\finding-lungs-in-ct-data-kaggle'
        self.root = root_path+"/datasets/finding-lungs-in-ct-data-kaggle/pngpic"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.img_paths = glob(self.root + r'\2d_images\*')
        # self.mask_paths = glob(self.root + r'\2d_masks\*')
        self.img_paths = glob(self.root + "/2d_images/*")
        self.mask_paths = glob(self.root + "/2d_masks/*")
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).convert('RGB').resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class KvasirSEGDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\codes\pytorch-nested-unet-master\pytorch-nested-unet-master\input\dsb2018_256'
        self.root = root_path+"/datasets/Kvasir-SEG"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.img_paths = glob(self.root + '\images\*')
        # self.mask_paths = glob(self.root + '\masks\*')
        self.img_paths = glob(self.root + '/images/*')
        self.mask_paths = glob(self.root + '/masks/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths,self.val_mask_paths  #因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class BUSIbenignDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\datasets\isbi'
        self.root = root_path+"/datasets/Dataset_BUSI_with_GT/benign"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # self.img_paths = glob(self.root + r'\train\images\*')
        # self.mask_paths = glob(self.root + r'\train\label\*')
        self.img_paths = glob(self.root + "/images/*")
        self.mask_paths = glob(self.root + "/masks/*")
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).convert('RGB').resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class BUSImalignantDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'E:\codes\pytorch-nested-unet-master\pytorch-nested-unet-master\input\dsb2018_256'
        self.root = root_path+"/datasets/Dataset_BUSI_with_GT/benign"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        n = len(os.listdir(self.root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "benign(%d).png" % (i + 1))  # liver is %03d
            mask = os.path.join(root, "benign(%d)_mask.png" % (i + 1))
            if not os.path.exists(img):
                continue
            pics.append(img)
            masks.append(mask)
        # self.img_paths = glob(self.root + '\images\*')
        # self.mask_paths = glob(self.root + '\masks\*')
        # self.img_paths = glob(self.root + '/images/*')
        # self.mask_paths = glob(self.root + '/masks/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths,self.val_mask_paths  #因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # pic = cv2.imread(pic_path)
        # mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = Image.open(pic_path).resize((224,224))
        mask = Image.open(mask_path).convert('L').resize((224,224))
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)