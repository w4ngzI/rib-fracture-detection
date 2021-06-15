from functools import partial
import torch
import torch.nn as nn
from torch import optim
from frac_dataset import TrainDataset
import transforms as tsfm
from metrics import dice, recall, precision, fbeta_score
from my_unet import U_Net
from losses import MixLoss, DiceLoss, FocalLoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

device = torch.device('cuda:0')
torch.cuda.set_device(0)

if __name__ == "__main__":
    train_image_dir = './train_image1'
    train_label_dir = './train_label1'
    val_image_dir = './train_image1'
    val_label_dir = './train_label1'

    batch_size = 4
    num_workers = 4
    num_samples = 4
    learning_rate = 1e-3
    best_val_loss = 1e9
    EPOCHES = 200

    #model = U_Net(1,1,16)
    model = U_Net()
    #model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = MixLoss(FocalLoss(), 0.2, DiceLoss(), 0.8)
    #criterion = Dice_loss()

    device_ids = [0,1]
    model = nn.DataParallel(model, device_ids = device_ids).cuda()

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    train_set = TrainDataset(train_image_dir, train_label_dir, num_samples = num_samples, transforms = transforms)
    val_set = TrainDataset(val_image_dir, val_label_dir, num_samples = num_samples, transforms = transforms)

    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = False,
                                    num_workers = num_workers, collate_fn = TrainDataset.collate_fn)
    val_dataloader = DataLoader(val_set, batch_size = batch_size, shuffle = False,
                                     num_workers = num_workers, collate_fn = TrainDataset.collate_fn)


    for epoch in range(EPOCHES):
        train_loss_per_epoch = 0
        train_dice_per_epoch = 0
        count = 0
        model.train()
        for i, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            image, label = data
            image = Variable(image).to(device)
            label = Variable(label).to(device)

            model.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss
            count += 1
            train_dice_per_epoch += dice(pred, label)
        train_loss_per_epoch /= count
        train_dice_per_epoch /= count

        val_loss_per_epoch = 0
        val_dice_per_epoch = 0
        count = 0
        model.eval()
        for i, data in enumerate(val_dataloader):
            image, label = data
            image = Variable(image).to(device)
            label = Variable(label).to(device)

            pred = model(image)
            loss = criterion(pred, label)
            val_loss_per_epoch += loss.item()
            count += 1
            val_dice_per_epoch += dice(pred, label)
        
        val_loss_per_epoch /= count
        val_dice_per_epoch /= count
        print('[epoch:%d] train_loss:%.5f val_loss:%.5f train_dice:%.5f val_dice:%.5f' \
            %(epoch+1, train_loss_per_epoch, val_loss_per_epoch, train_dice_per_epoch, val_dice_per_epoch))
        
        
        if val_loss_per_epoch < best_val_loss:
            best_val_loss = val_loss_per_epoch
            torch.save(model.state_dict(),'bestmodel' + '.pth')
            print('find better model at epoch %d with val_loss: %.5f' %(epoch+1, val_loss_per_epoch))

