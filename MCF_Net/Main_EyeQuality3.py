import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time


from progress.bar import Bar
import torchvision.transforms as transforms
from mydataloader.EyeQQQ_loader import DatasetGenerator1
from utils.trainer3 import train_step, validation_step, save_output
from utils.metric import compute_metric
from networks.Discriminator import Discriminator
import pandas as pd
from networks.densenet_mcf import dense121_mcs

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

data_root = '../Kaggle_DR_dataset/'

# Setting parameters
parser = argparse.ArgumentParser(description='EyeQ_dense121')
parser.add_argument('--model_dir', type=str, default='./result/')
parser.add_argument('--save_dir', type=str, default='./save/')
parser.add_argument('--pre_model', type=str, default='DenseNet121_v3_v1')
parser.add_argument('--save_model', type=str, default='DenseNet121_v3_v1')
parser.add_argument('--crop_size', type=int, default=224)
#--------------------------------------------------
parser.add_argument('--label_idx', type=list, default=['Good', 'Reject'])
#--------------------------------------------------
parser.add_argument('--n_classes', type=int, default=2)
# Optimization options
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)

args = parser.parse_args()

# Images Labels
train_images_dir = data_root + '/EyeQ-194'
label_train_file = '../data/Label_EyeQ_194.csv'
test_images_dir = data_root + '/DRIMDB'
label_test_file = '../data/Label_DRIMDB.csv'

target_images_dir = data_root + '/DRIMDB'
target_test_file = '../data/Label_DRIMDB.csv'

save_file_name = args.model_dir + args.save_model + '.csv'

best_metric = np.inf
best_iter = 0
# options
cudnn.benchmark = True

model = dense121_mcs(n_class=args.n_classes) # 教师网络
model2 = dense121_mcs(n_class=args.n_classes) # 学生网络
discriminator = Discriminator() # 判别器

model.to(device)
model2.to(device)
discriminator.to(device)


# if args.pre_model is not None:
#     loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
#     model.load_state_dict(loaded_model['state_dict'])

criterion = torch.nn.BCELoss(reduction='mean')


optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
st_optimizer = torch.optim.SGD(model2.parameters(), lr=0.1)
dis_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1)

print('teach_Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
print('st_Total params: %.2fM' % (sum(p.numel() for p in model2.parameters()) / 1000000.0))
print('dis_Total params: %.2fM' % (sum(p.numel() for p in discriminator.parameters()) / 1000000.0))

transform_list1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, +180)),
    ])

transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

# 修改了一下 可以使用方便跨域测试
data_train = DatasetGenerator1(data_dir=train_images_dir, list_file=label_train_file,dataset="EyeQ", transform1=transform_list1,
                              transform2=transformList2, n_class=args.n_classes, set_name='train')
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

data_test = DatasetGenerator1(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                             transform2=transformList2, n_class=args.n_classes,dataset="DRIMDB", set_name='EyeQ')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)

target_train = DatasetGenerator1(data_dir=target_images_dir, list_file=target_test_file,dataset="DRIMDB", transform1=transform_list1,
                              transform2=transformList2, n_class=args.n_classes, set_name='train')
target_loader = torch.utils.data.DataLoader(dataset=target_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

# label_train = DatasetGenerator(data_dir=target_images_dir, list_file=target_test_file,dataset="DRIMDB", transform1=transform_list1,
#                               transform2=transformList2, n_class=args.n_classes, set_name='train')
# label_loader = torch.utils.data.DataLoader(dataset=label_train, batch_size=args.batch_size,
#                                                shuffle=True, num_workers=4, pin_memory=True)
# Train and val
# for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
#     imagesA = imagesA.cuda()
#     imagesB = imagesB.cuda()
#     imagesC = imagesC.cuda()
#
#     begin_time = time.time()
#     _, _, _, _, result_mcs ,_= model2(imagesA, imagesB, imagesC)
#     outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
#     batch_time = time.time() - begin_time
def test(model, target_test_loader):
    model.eval()
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for (imagesA, imagesB, imagesC,labels) in target_test_loader:
            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
            imagesC = imagesC.cuda()
            labels = labels.cuda()
            _, _, _, _, result_mcs ,_= model(imagesA, imagesB, imagesC)

            for i in range(result_mcs.shape[0]):
                left = result_mcs[i][0]
                right = result_mcs[i][1]
                if left > right :
                    result_mcs[i][0] = 1
                    result_mcs[i][1] = 0
                else:
                    result_mcs[i][0] = 0
                    result_mcs[i][1] = 1

            correct += torch.sum(result_mcs == labels)/2
    acc = 100. * correct / len_target_dataset
    return acc
best_acc = 0
stop = 0
log = []
for epoch in range(0, args.epochs):
    p = 0.01 * epoch

    _ = train_step(train_loader,target_loader, model,model2,discriminator, epoch, optimizer,st_optimizer,dis_optimizer, criterion, args,p)
    # Test
    stop += 1
    teacher_acc = test(model, target_loader)
    student_acc = test(model2, target_loader)
    info = 'teacher_acc {:4f} | student_acc {:4f} '.format(teacher_acc,student_acc)
    np_log = np.array(log, dtype=float)
    np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
    if best_acc < student_acc:
        best_acc = student_acc
    # if student_acc > 90 :
    #     break
    print(info)
print('best result: {:.4f}'.format(best_acc))

# Testing teacher
outPRED_mcs = torch.FloatTensor().cuda()
model.eval()
iters_per_epoch = len(test_loader)
print("iters_per_epoch:",iters_per_epoch)
bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
bar.check_tty = False
for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
    imagesA = imagesA.cuda()
    imagesB = imagesB.cuda()
    imagesC = imagesC.cuda()

    begin_time = time.time()
    _, _, _, _, result_mcs ,_= model(imagesA, imagesB, imagesC)
    outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                           batch_time=batch_time * (iters_per_epoch - epochID) / 60)
    bar.next()
bar.finish()

# save result into excel:
save_output(label_test_file, outPRED_mcs, args, save_file=save_file_name)

#--------------------------------------------------
# evaluation:
df_gt = pd.read_csv(label_test_file)
img_list = df_gt["image"].tolist()
GT_QA_list = np.array(df_gt["quality"].tolist())
img_num = len(img_list)
label_list = ['Good', 'Reject']

df_tmp = pd.read_csv(save_file_name)
print(df_tmp)
predict_tmp = np.zeros([img_num, 2])
for idx in range(2):
    predict_tmp[:, idx] = np.array(df_tmp[label_list[idx]].tolist())
tmp_report = compute_metric(GT_QA_list, predict_tmp, target_names=label_list)

print(' Accuracy: ' + str("{:0.4f}".format(np.mean(tmp_report['Accuracy']))) +
      ' Precision: ' + str("{:0.4f}".format(np.mean(tmp_report['Precision']))) +
      ' Sensitivity: ' + str("{:0.4f}".format(np.mean(tmp_report['Sensitivity']))) +
      ' F1: ' + str("{:0.4f}".format(np.mean(tmp_report['F1']))))

# Testing
outPRED_mcs = torch.FloatTensor().cuda()
model2.eval()
iters_per_epoch = len(test_loader)
print("iters_per_epoch:",iters_per_epoch)
bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
bar.check_tty = False
for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
    imagesA = imagesA.cuda()
    imagesB = imagesB.cuda()
    imagesC = imagesC.cuda()

    begin_time = time.time()
    _, _, _, _, result_mcs ,_= model2(imagesA, imagesB, imagesC)
    outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                           batch_time=batch_time * (iters_per_epoch - epochID) / 60)
    bar.next()
bar.finish()

# save result into excel:
save_output(label_test_file, outPRED_mcs, args, save_file=save_file_name)

#--------------------------------------------------
# evaluation:
df_gt = pd.read_csv(label_test_file)
img_list = df_gt["image"].tolist()
GT_QA_list = np.array(df_gt["quality"].tolist())
img_num = len(img_list)
label_list = ['Good', 'Reject']

df_tmp = pd.read_csv(save_file_name)
print(df_tmp)
predict_tmp = np.zeros([img_num, 2])
for idx in range(2):
    predict_tmp[:, idx] = np.array(df_tmp[label_list[idx]].tolist())
tmp_report = compute_metric(GT_QA_list, predict_tmp, target_names=label_list)

print(' Accuracy: ' + str("{:0.4f}".format(np.mean(tmp_report['Accuracy']))) +
      ' Precision: ' + str("{:0.4f}".format(np.mean(tmp_report['Precision']))) +
      ' Sensitivity: ' + str("{:0.4f}".format(np.mean(tmp_report['Sensitivity']))) +
      ' F1: ' + str("{:0.4f}".format(np.mean(tmp_report['F1']))))
