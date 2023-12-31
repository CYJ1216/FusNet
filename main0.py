
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from model.UNet import Unet,resnet34_unet
from model.attention_unet import AttU_Net
from model.channel_unet import myChannelUnet
from model.r2unet import R2U_Net
from model.segnet import SegNet
from model.unetpp import NestedUNet
from model.fcn import get_fcn8s
from model.cgnet import Context_Guided_Network
from torchvision.models import vgg16
from model.fusNet import FusNet
from model.fusNet_res2 import FusNet_res2
from model.fusNet_swin import FusNet_swin
from model.fusNet_ndiv import FusNet_ndiv
from model.fusNet_add import FusNet_add
from dataset import *
from new_metric import get_iou,get_pre,get_dice
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot

# from cenet import CE_Net_

from loss import FocalLoss
from loss1 import bce_ssim_loss
from loss1 import IOU


# from cenet import CE_Net_

#更改torch.cuda.OutOfMemoryError: CUDA out of memory
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# OSError: image file is truncated (59 bytes not processed)
# 传入的图片已经超过了MAXBLOCK限制的大小，PIL处理不了，必须要把这个图片删除一部分
# 把LOAD_TRUNCATED_IMAGES设为true，实际上会导致你加载的图片少掉一部分，虽然在大数据中，偶尔一两张被裁掉没什么大问题，但是还是应当注意，不要暴力用才好。
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
root_path='/home/user_home/chengyingjie/project2'

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=15)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='FusNet_add',
                       help='UNet/resnet34_unet/unet++/cgnet/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s/FusNet/FusNet_res2/FusNet_swin/FusNet_ndiv/FusNet_add')
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument('--dataset', default='BUSIbenign',  # dsb2018_256
                       help='dataset name:BUSIbenign/KvasirSEG/liver/ichallenge/cvc300/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    # parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--log_dir", default=root_path+'/result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    if  args.arch == 'cgnet':
        model = Context_Guided_Network(1).to(device)
    if args.arch =='FusNet':
        model = FusNet().to(device)
    if args.arch =='FusNet_res2':
        model = FusNet_res2().to(device)
    if args.arch =='FusNet_swin':
        model = FusNet_swin().to(device)
    if args.arch == 'FusNet_ndiv':
        model = FusNet_ndiv().to(device)
    if args.arch == 'FusNet_add':
        model = FusNet_add().to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1,pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args,3,1).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(3,1).to(device)
    if args.arch == 'segnet':
        model = SegNet(3,1).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3,1).to(device)
    # if args.arch == 'fcn32s':
    #     model = get_fcn32s(1).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(3,1).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset !='esophagus' ,"fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
        model = get_fcn8s(1).to(device)
    if args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset =='KvasirSEG':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = KvasirSEGDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = KvasirSEGDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =='liver':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =='ichallenge':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = ichallengeDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = ichallengeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =='cvc300':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = cvc300Dataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = cvc300Dataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =="esophagus":
        train_dataset = esophagusDataset(r"train", transform=x_transforms,target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = esophagusDataset(r"val", transform=x_transforms,target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset == "dsb2018Cell":
        train_dataset = dsb2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = dsb2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset == 'corneal':
        train_dataset = CornealDataset(r'train',transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'driveEye':
        train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'isbiCell':
        train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'kaggleLung':
        train_dataset = LungKaggleDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = LungKaggleDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = LungKaggleDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'BUSIbenign':
        train_dataset = BUSIbenignDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = BUSIbenignDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = BUSIbenignDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        dice_total = 0
        acc_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            dice_total += get_dice(mask[0], img_y)
            miou_total += get_iou(mask[0],img_y)  #获取当前预测图的miou，并加到总miou中
            acc_total += get_pre(mask[0],img_y)
            if i < num:i+=1   #处理验证集下一张图
        aver_iou = miou_total / num
        aver_dice = dice_total / num
        aver_acc = acc_total/num
        print('miou=%f,aver_dice=%f,aver_acc=%f' % (aver_iou,aver_dice,aver_acc))
        logging.info('miou=%f,aver_dice=%f,aver_acc=%f' % (aver_iou,aver_dice,aver_acc))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            # torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
            torch.save(model.state_dict(), root_path+'/saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch) + '.pth')
        return best_iou,aver_iou,aver_acc,aver_dice

def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_acc,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    acc_list = []
    dice_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    # loss += bce_ssim_loss(output, labels)     #这个进行损失函数的对比实验
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output, labels)
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou,aver_iou,aver_acc,aver_dice = val(model,best_iou,val_dataloader)
        iou_list.append(aver_iou)
        acc_list.append(aver_acc)
        dice_list.append(aver_dice)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f best_iou:%0.3f" % (epoch, epoch_loss,best_iou))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&acc',iou_list, acc_list)
    metrics_plot(args,'hd',dice_list)
    return model

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        # dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        dir = os.path.join(root_path+'/saved_predict', str(args.arch), str(args.batch_size), str(args.epoch), str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    # model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.load_state_dict(torch.load(root_path + '/saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch) + '.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        dice_total = 0
        acc_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(mask_path[0], predict)
            acc = get_pre(mask_path[0],predict)
            acc_total += acc

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            #print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            #print(mask_path[0])
            # if save_predict == True:
            #     if args.dataset == 'driveEye':
            #         saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
            #         saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
            #         plt.savefig(saved_predict)
            #     else:   #mask_path[0].split('\\')[-1]
            #         plt.savefig(dir +'/'+ os.path.basename(mask_path[0]))
            if save_predict == True:
                #mask_path[0].split('\\')[-1]
                plt.savefig(dir +'/'+ os.path.basename(mask_path[0]))
            #plt.pause(0.01)
            print('iou={},acc={}'.format(iou,acc))
            if i < num:i+=1   #处理验证集下一张图
        #plt.show()
        print('miou=%f,aver_dice=%f,acc=%f' % (miou_total/num,dice_total/num,acc_total/num))
        logging.info('miou=%f,aver_dice=%f,acc=%f' % (miou_total/num,dice_total/num,acc_total/num))
        #print('M_dice=%f' % (dice_total / num))

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms =transforms.ToTensor()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = try_gpu(3)
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss()
    # criterion= IOU()
    optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)