from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.model import PointNetCls, feature_transform_regularizer
from residual_transformer.model import ResidualTransformer
from dataset.dynamic_dataset import DynamicModelNetDataset
from dataset.static_dataset import StaticModelNetDataset
import torch.nn.functional as F
from tqdm import tqdm

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument(
        '--nepoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='..\..\data\ModelNet40_numpy\\', help="dataset path")
    parser.add_argument('--static_dataset', action='store_true', help='use static point sampling for each epoch')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    print("Cuda Available: ", torch.cuda.is_available(), " | current device: ", device)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # load dataset

    # if dynamic, use DynamicModelNetDataset
    if (opt.static_dataset):
        print("Using Static Dataset...")
        dataset = StaticModelNetDataset(
            root_dir=opt.dataset,
            device=device,
            folder='train')

        test_dataset = StaticModelNetDataset(
            root_dir=opt.dataset,
            device=device,
            folder='test')

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True)

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True)
    else:
        print("Using Dynamic Dataset...")
        dataset = DynamicModelNetDataset(
            root_dir=opt.dataset,
            npoints=opt.num_points,
            folder='train')

        test_dataset = DynamicModelNetDataset(
            root_dir=opt.dataset,
            folder='test',
            npoints=opt.num_points)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            pin_memory = True)

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            pin_memory = True)

    print(len(dataset), len(test_dataset))
    num_classes = 40
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    classifier = ResidualTransformer(k=num_classes)
    # classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    
    classifier.to(device)

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.float().transpose(2, 1)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            # pred, trans, trans_feat = classifier(points)
            pred = classifier(points)
            loss = F.nll_loss(pred, target)
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.float().transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                # pred, _, _ = classifier(points)
                pred = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
        
        scheduler.step()

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.float().transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        # pred, _, _ = classifier(points)
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))

if __name__ == '__main__':
    train()