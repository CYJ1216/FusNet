# FusNet
FusNet: A Deep Information Fusion Network for Medical Image Segmentation

两个预加载模型：（.pth）(大于100M放在pth文件夹中)
1，res2net50_v1b_26w_4s-3cf99910.pth
注意更改fusnet/model/res2net.py中res2net50_v1b_26w_4s的读入位置
2，swin_tiny_patch4_window7_224.pth
注意更改fusnet/model/swin.py中swin_tiny_patch4_window7_224的读入位置

各个文档的含义说明：
1、model文件夹：各个模型的所有代码，包括预训练模型res2net50_v1b_26w_4s-3cf99910.pth ，swin_tiny_patch4_window7_224.pth。
这些模型主要是基于Unet的以及其变体的模型，同时还包含了fusnet；
2、result：每次运行完一个模型后，都会将模型的训练日志写进去，但是这种日志每一步的loss损失和指标，plot是模型的训练图、指标图、损失图；
3、saved_model:保存的指标最高的模型，在test的时候会根据这个模型来进行推理；
4、saved_predict：将原图，预测图（中间的），和真值标签图进行比较的图；
5、结果比较：里面是每个数据集下模型的性能指标图，这些指标请用训练过程中指标最高的那个。
6、loss.py、loss1.py：测试的损失函数代码；最终用的是BCELoss。
7、new_metric.py：测试Miou、ACC（准确率）和Dice指标。
8、main.py：启动代码，里面设置用什么模型、loss设置、batchsize等；
9、dataset.py：读取数据集，不同数据集有不同样式，这里设置数据的路径，以及读取样式。
各数据集都是我处理后的呈现出来的。（datasets文件夹放入自己需要跑的图片。本项目用的数据集是Kvasir-SEG , Ichallenge-PM , dsb2018cell ,BUSI）


跑代码时候，需要调整：
（1）文件中各个文件在电脑的读入位置。使用的是绝对路径，要调整。统一根目录放在每个文件开头，记得修改。
（2）main.py 设置用什么模型、损失函数等。
（3）dataset.py 设置数据集路径和读取样式。读取样式要稍微改一改，不同的数据样式会稍微不同。
