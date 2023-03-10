import time
import torch
from progress.bar import Bar
import numpy as np
import pandas as pd
import math
from itertools import zip_longest

def train_step(train_loader, target_loader,model,st_model,dis_model, epoch, optimizer,st_optimizer,dis_optimizer, criterion, args,p):

    # switch to train mode
    model.train()
    st_model.train()
    dis_model.train()

    epoch_loss = 0.0
    loss_w =args.loss_w

    iters_per_epoch = len(train_loader) # 一次遍历整个数据所要的次数
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False
    step = 0
# TODO 修改dataset
    for  (imagesA, imagesB, imagesC, labels),(t_imagesA, t_imagesB, t_imagesC, t_labels) in zip_longest(train_loader,target_loader):
        start_time = time.time()
        step = step + 1
        torch.set_grad_enabled(True)

        # 1.G1 的loss
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()
        labels = labels.cuda()
        out_A, out_B, out_C, out_F, combine, feature_all = model(imagesA, imagesB, imagesC)
        loss_x = criterion(out_A, labels)

        loss_y = criterion(out_B, labels)
        loss_z = criterion(out_C, labels)
        loss_c = criterion(out_F, labels)
        loss_f = criterion(combine, labels)
        lossValue = loss_w[0] * loss_x + loss_w[1] * loss_y + loss_w[2] * loss_z + loss_w[3] * loss_c + loss_w[
            4] * loss_f
        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()


        t_imagesA = t_imagesA.cuda()
        t_imagesB = t_imagesB.cuda()
        t_imagesC = t_imagesC.cuda()
        t_labels = t_labels.cuda()

        t_out_A, t_out_B, t_out_C, t_out_F, t_combine, t_feature_all = model(t_imagesA, t_imagesB, t_imagesC) # 教师模型输出的伪标签


        # 2. D的loss
        # 2.1 判真
        dis_optimizer.zero_grad()
        real_output = dis_model(feature_all.detach()) # [4,2]
        # print("real_output",real_output)

        d_real_loss = criterion(real_output,
                                torch.ones_like(real_output))
        d_real_loss.backward()

        fake_output = dis_model(t_feature_all.detach())  # [4,2]
        # print("fake_output",fake_output)

        d_fake_loss = criterion(fake_output,
                                torch.zeros_like(fake_output))
        d_fake_loss.backward()
        # -------------------------------------------  需要把真假loss加起来后再step吗
        d_loss = d_real_loss + d_fake_loss
        dis_optimizer.step()

        st_out_A, st_out_B, st_out_C, st_out_F, st_combine, _ = st_model(t_imagesA, t_imagesB, t_imagesC)
        # 每5次学习一次
        # 3 G2 loss
        # TODO
        T = 1 / (1 + math.exp(-10 * p))
        pseudo_label = torch.ones_like(t_combine)

        st_label = torch.zeros_like(st_combine)
        # 构建 学生  hard标签
        for i in range(st_combine.shape[0]):
            if st_combine[i][0] > st_combine[i][1]:
                st_label[i][0] = 1
            else:
                st_label[i][1] = 1

        # 构建 教师  hard标签
        t_label = torch.zeros_like(t_combine)
        for i in range(t_combine.shape[0]):
            if t_combine[i][0] > t_combine[i][1]:
                t_label[i][0] = 1
            else:
                t_label[i][1] = 1

        for i in range(st_combine.shape[0]):
            t_max = torch.max(t_combine[i][0], t_combine[i][1])
            st_max = torch.max(st_combine[i][0], st_combine[i][1])
            Min = min(st_max, T)
            if t_max > Min:  # 大于则选取老师label
                pseudo_label[i] = t_label[i]
            else:  # 小于则选取学生label
                pseudo_label[i] = st_label[i]
        # 第一次保存伪标签
        if epoch == 0 :
            t_label = pseudo_label

        # 每5次更新一次伪标签
        if epoch%5 == 0:
            t_label = pseudo_label


        # print("t_combine",t_combine)
        # print("st_combine",st_combine)
        # print("pseudo_label",pseudo_label)
        st_loss_x = criterion(st_out_A, t_label)
        st_loss_y = criterion(st_out_B, t_label)
        st_loss_z = criterion(st_out_C, t_label)
        st_loss_c = criterion(st_out_F, t_label)
        st_oss_f = criterion(st_combine, t_label)
        st_lossValue = loss_w[0] * st_loss_x + loss_w[1] * st_loss_y + loss_w[2] * st_loss_z + loss_w[
            3] * st_loss_c + \
                       loss_w[
                           4] * st_oss_f

        st_optimizer.zero_grad()
        st_lossValue.backward()
        st_optimizer.step()
        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time=batch_time*(iters_per_epoch-step)/60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss


def validation_step(val_loader, model, criterion):

    # switch to train mode
    model.eval()
    epoch_loss = 0
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()
        labels = labels.cuda()

        _, _, _, _, outputs,_ = model(imagesA, imagesB, imagesC)
        with torch.no_grad():
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    return epoch_loss


def save_output(label_test_file, dataPRED, args, save_file):
    label_list = args.label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["image"].tolist()

    # list=[表达式 for 变量 in 范围 if 条件]
    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['image_name'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['image_name']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)
    print(save_file)
    print("yes")


