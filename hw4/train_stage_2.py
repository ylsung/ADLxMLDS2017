from utils import SimpleDataset, np2Var, tensor2Var, create_fake_tags, array_back_style
from model import Generator, Critic, Generator_img2img, Critic_img2img

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def calc_grad_penalty(args, net, real_data, fake_data, real_tags, fake_tags):

    alpha = torch.FloatTensor(real_data.size(0), 1, 1, 1).uniform_(0, 1)
    # alpha.uniform_(0, 1)

    # alpha_t = alpha.expand(real_data.size())

    inter_data = alpha * real_data + ((1 - alpha) * fake_data)

    inter_tags = alpha * real_tags + ((1 - alpha) * fake_tags)

    # inter_tags = real_tags
    inter_data = tensor2Var(inter_data, requires_grad=True)
    inter_tags = tensor2Var(inter_tags, requires_grad=True)

    disc_interpolates = net(inter_data, inter_tags)

    # assert disc_interpolates.size() == one_label.size(), 'one_label size mismatch'
    input_data = [inter_data, inter_tags]
    input_one = [tensor2Var(torch.ones(disc_interpolates.size())), 
    tensor2Var(torch.ones(disc_interpolates.size()))]

    # input_data = inter_data
    # input_one = tensor2Var(torch.ones(disc_interpolates.size()))

    gradients_x, gradients_tags = grad(
        outputs=disc_interpolates, 
        inputs=input_data,
        grad_outputs=input_one,
        create_graph=True, retain_graph=True, only_inputs=True)

    while len(gradients_x.size()) > 1:
        gradients_x = gradients_x.norm(2, dim=(len(gradients_x.size()) - 1))

    gradients_tags = gradients_tags.norm(2, dim=1)

    gradients = (gradients_x ** 2 + gradients_tags ** 2).sqrt()
    # gradients = gradients_x
    gradient_penalty = args.LAMBDA * ((gradients - 1.0) ** 2).mean()
    return gradient_penalty

def calc_x_grad_penalty(args, net, real_data, fake_data, real_tags, fake_tags):

    alpha = torch.FloatTensor(real_data.size(0), 1, 1, 1).uniform_(0, 1)
    # alpha.uniform_(0, 1)

    # alpha_t = alpha.expand(real_data.size())

    inter_data = alpha * real_data + ((1 - alpha) * fake_data)


    # inter_tags = real_tags
    inter_data = tensor2Var(inter_data, requires_grad=True)

    disc_interpolates, _ = net(inter_data, real_tags)

    # assert disc_interpolates.size() == one_label.size(), 'one_label size mismatch'

    input_data = inter_data
    input_one = tensor2Var(torch.ones(disc_interpolates.size()))

    gradients_x = grad(
        outputs=disc_interpolates, 
        inputs=input_data,
        grad_outputs=input_one,
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    while len(gradients_x.size()) > 1:
        gradients_x = gradients_x.norm(2, dim=(len(gradients_x.size()) - 1))

    gradients = gradients_x
    gradient_penalty = args.LAMBDA * ((gradients - 1.0) ** 2).mean()
    return gradient_penalty
def bce_loss(output, target, mask):
    assert output.size() == target.size()
    assert output.size() == mask.size()
    return -(target * torch.log(output + 1e-8) * mask + \
        (1.0 - target) * torch.log(1.0 - output + 1e-8) * mask).sum(dim=1).mean()

def train_C(args, data, tags, mask, netG_pre, netG, netC, optC):

    noise = torch.FloatTensor(data.size(0), args.nz, 1, 1).uniform_(0, args.noise_max)

    noise_v = tensor2Var(noise, volatile=True)
    data_v, tags_v, mask_v = tensor2Var(data), tensor2Var(tags), tensor2Var(mask)

    _, tags_id = torch.topk(tags, 2, dim=1)
    fake_tags = torch.zeros(tags.size())
    for i in range(tags_id.size(0)):
        hair, eyes = create_fake_tags(tags_id[i, 0], tags_id[i, 1])
        fake_tags[i, hair] = 1.0
        fake_tags[i, eyes] = 1.0
    fake_tags_v = tensor2Var(fake_tags)

    fake_data_v = netG_pre(noise_v, tags_v)
    fake_data_v = Variable(netG(fake_data_v, tags_v).data)

    # real data with real tags
    real_real, real_real_class = netC(data_v, tags_v)
    # real data with fake tags
    real_fake, real_fake_class = netC(data_v, fake_tags_v)
    # fake data with real tags
    fake_real, fake_real_class = netC(fake_data_v, tags_v)
    # fake data with fake tags
    # fake_fake = netC(fake_data_v, fake_tags_v)

    # real_fake_grads = calc_grad_penalty(args, netC, data, data, tags, fake_tags)
    fake_real_grads = calc_x_grad_penalty(args, netC, data, fake_data_v.cpu().data, tags, tags)
    # fake_fake_grads = calc_grad_penalty(args, netC, data, fake_data_v.cpu().data, tags, fake_tags)

    real = real_real.mean()
    fake = fake_real.mean()
    grads_penalty = fake_real_grads

    class_loss = (bce_loss(real_real_class, tags_v, mask_v) + \
        bce_loss(fake_real_class, tags_v, mask_v)) / 2.0


    netC.zero_grad()


    loss = -real + fake + grads_penalty + class_loss
    loss.backward()
    # grads_penalty.backward()


    optC.step()

    return (real - fake).cpu().data.numpy()[0], class_loss.cpu().data.numpy()[0]

def train_G(args, data, tags, mask, netG_pre, netG, netC, optG):
    noise = torch.FloatTensor(data.size(0), args.nz, 1, 1).uniform_(0, args.noise_max)

    noise_v = tensor2Var(noise)

    # fake_tags = torch.zeros(tags.size())
    # for i in range(fake_tags.size(0)):
    #     hair, eyes = create_fake_tags(0, 13)
    #     fake_tags[i, hair] = 1.0
    #     fake_tags[i, eyes] = 1.0
    # fake_tags_v = tensor2Var(fake_tags)

    tags_v, mask_v = tensor2Var(tags), tensor2Var(mask)

    fake_data_v = Variable(netG_pre(noise_v, tags_v).data)

    fake_data_v = netG(fake_data_v, tags_v)

    fake_real, fake_real_class = netC(fake_data_v, tags_v)

    loss = -fake_real.mean() + bce_loss(fake_real_class, tags_v, mask_v)

    netG.zero_grad()
    loss.backward()

    optG.step()

    return (-loss).cpu().data.numpy()[0]


def run_epoch(args, tr_loader, netG_pre, netG, netC, optG, optC):
    # for i, (data, label) in enumerate(tr_loader):
    data_iter = iter(tr_loader)
    iteration = 0

    while iteration < len(data_iter):
        ################
        ## update Critic
        ################
        for p in netG.parameters():
            p.requires_grad = False
        for p in netC.parameters():
            p.requires_grad = True
        j = 0
        while iteration < len(data_iter) and j < 5:
            data, tags, mask = next(data_iter)
            W, C = train_C(args, data, tags, mask, netG_pre, netG, netC, optC)

            # for p in netC.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            j += 1
            iteration += 1
        ###################
        ## update Generator
        ###################
        for p in netG.parameters():
            p.requires_grad = True
        for p in netC.parameters():
            p.requires_grad = False

        F = train_G(args, data, tags, mask, netG_pre, netG, netC, optG)
    
    return W, C, F, data, tags.numpy(), mask.numpy()

def train(args, logger, tags_dict, mask_dict, id2style):

    # tr_dset = SimpleDataset(args.img_dir, tags_dict, mask_dict, transform)
    tr_dset = SimpleDataset(args.img_dir, tags_dict, mask_dict, args.image, args.degree)

    tr_loader = DataLoader(
        tr_dset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=16,
        )

    # netG = G_net(args.nz, len(id2style))
    # netC = D_net(len(id2style))

    netG_pre = Generator(
        nz=args.nz,
        nc=args.nc,
        ntext=len(id2style),
        dim=64,
        image_size=64
        )

    netG = Generator_img2img(
        nz=args.nc,
        nc=args.nc,
        ntext=len(id2style),
        dim=64,
        image_size=args.image
        )
    netC = Critic_img2img(
        nz=args.nz,
        nc=args.nc,
        ntext=len(id2style),
        dim=64,
        image_size=args.image
        )
    if torch.cuda.is_available():
        netG_pre = netG_pre.cuda()
        netG, netC = netG.cuda(), netC.cuda()


    netG_pre_path = 'ADLxMLDS_hw4_model/binary_style_mask_aug/netG_500.pth'
    netG_pre.load_state_dict(torch.load(netG_pre_path, map_location=lambda storage, loc: storage))
    for p in netG_pre.parameters():
        p.requires_grad = False
    netG_pre.eval()


    logger.info(netG)
    logger.info(netC)

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optC = torch.optim.Adam(netC.parameters(), lr=args.lr, betas=(0.5, 0.9))

    min_W = 1000.0
    sample_size = 16

    save_folder = os.path.join(args.root, args.save)
    load_folder = os.path.join(args.root, args.load)

    if args.load != '':
        netG_path = os.path.join(load_folder, 'netG_%d.pth' % args.model_id)
        netC_path = os.path.join(load_folder, 'netC_%d.pth' % args.model_id)
        netG.load_state_dict(torch.load(netG_path, map_location=lambda storage, loc: storage))
        netC.load_state_dict(torch.load(netC_path, map_location=lambda storage, loc: storage))

        logger.info('load from: %s success!!!!!!!!!!!!!' % netG_path)
        logger.info('load from: %s success!!!!!!!!!!!!!' % netC_path)


    W_list = []
    C_list = []
    epoch_list = []
    w2epoch = os.path.join(save_folder, 'w2epoch.png')
    c2epoch = os.path.join(save_folder, 'c2epoch.png')


    noise = torch.FloatTensor(sample_size, args.nz, 1, 1).uniform_(0, args.noise_max)
    noise_v = tensor2Var(noise, volatile=True)

    fake_tags = torch.zeros(sample_size, len(id2style))
    for i in range(fake_tags.size(0)):
        hair, eyes = create_fake_tags(-1, -1)
        fake_tags[i, hair] = 1.0
        fake_tags[i, eyes] = 1.0
        logger.info('%d: %s, %s' % (i, id2style[hair], id2style[eyes]))
    fake_tags_v = tensor2Var(fake_tags, volatile=True)

    print_ground_truth = 1

    transform = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Scale(64),
        tv.transforms.ToTensor(),
        ])

    for epoch in range(args.max_epoch  + 1):
        W, C, _, real_data, real_tags, real_mask = run_epoch(args, tr_loader, netG_pre, 
            netG, netC, optG, optC)

        logger.info('epoch: %d, Wasserstain Dist: %.4f, Class loss: %.4f' % (epoch, W, C))

        fake_data_v = netG_pre(noise_v, fake_tags_v)
        fake_data_v = netG(fake_data_v, fake_tags_v)

        if print_ground_truth:
            choose = random.randint(0, real_tags.shape[0] - sample_size)
            real_data = real_data[choose: choose+sample_size]
            real_tags = real_tags[choose: choose+sample_size]
            real_mask = real_mask[choose: choose+sample_size]

            real_style = array_back_style(real_tags, id2style)

            logger.info('real tags')
            logger.info(real_style)
            logger.info(real_mask)
            print_ground_truth = 0
            img_path = os.path.join(save_folder, 'real.png')
            tv.utils.save_image(real_data, img_path, nrow=4)

        img_list = []
        for i in range(fake_data_v.size(0)):
            img_list.append(transform(fake_data_v.cpu().data[i]))
        img_data = torch.stack(img_list)

        img_path = os.path.join(save_folder, 'fake_%d.png' % epoch)
        tv.utils.save_image(img_data, img_path, nrow=4)

        W_list.append(W)
        C_list.append(C)
        epoch_list.append(epoch)

        # plot(epoch_list, W_list, 'epochs', 'Wasserstain Distance', w2epoch)
        # plot(epoch_list, C_list, 'epochs', 'Class loss', c2epoch)

        if epoch % 50 == 0 and epoch != 0:
            netG_path = os.path.join(save_folder, 'netG_%d.pth' % epoch)
            netC_path = os.path.join(save_folder, 'netC_%d.pth' % epoch)

            torch.save(netG.state_dict(), netG_path)
            torch.save(netC.state_dict(), netC_path)

def test(args, logger, _id, _tags, id2style):
    netG_pre = Generator(
        nz=args.nz,
        nc=args.nc,
        ntext=len(id2style),
        dim=64,
        image_size=64
        )
    netG = Generator_img2img(
        nz=args.nc,
        nc=args.nc,
        ntext=len(id2style),
        dim=64,
        image_size=args.image
        )

    if torch.cuda.is_available():
        netG = netG.cuda()
        netG_pre = netG_pre.cuda()

    logger.info(netG_pre)
    logger.info(netG)

    netG_pre_path = 'ADLxMLDS_hw4_model/binary_style_mask_aug/netG_500.pth'
    load_folder = os.path.join(args.root, args.load)

    if args.load != '':
        netG_path = os.path.join(load_folder, 'netG_%d.pth' % args.model_id)
        
        netG.load_state_dict(torch.load(netG_path, map_location=lambda storage, loc: storage))
        netG_pre.load_state_dict(torch.load(netG_pre_path, map_location=lambda storage, loc: storage))

        logger.info('load from: %s success!!!!!!!!!!!!!' % netG_path)
        logger.info('load from: %s success!!!!!!!!!!!!!' % netG_pre_path)
    else:
        logger.info('please load a model!!!!!')
        exit()

    i = 0
    torch.manual_seed(307)

    img_dir = os.path.join(args.img_save)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


    transform = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Scale(64),
        tv.transforms.ToTensor(),
        ])
    netG.eval()
    netG_pre.eval()
    while i * args.batch < _tags.shape[0]:
        ba_tags = _tags[i*args.batch: (i+1)*args.batch]
        ba_id = _id[i*args.batch: (i+1)*args.batch]
        ba_tags_v = np2Var(ba_tags, volatile=True)
        # img_list = []
        for s in range(1, args.sample_num+1):

            noise = torch.FloatTensor(ba_tags.shape[0], args.nz , 1, 1).uniform_(0, args.noise_max)

            noise_v = tensor2Var(noise, volatile=True)

            fake_img = netG_pre(noise_v, ba_tags_v)
            fake_img = netG(fake_img, ba_tags_v).cpu().data

            # fake_img = transform(fake_img[0])

            # img_list.append(fake_img)
            for j in range(ba_tags.shape[0]):
                img = fake_img[j]
                img = transform(img)
                img_name = os.path.join(img_dir, 'sample_%s_%d.jpg' % (ba_id[j], s))
                tv.utils.save_image(img, img_name)
        # img = torch.stack((img_list))
        # img_name = os.path.join(img_dir, 'sample_%d.jpg' % i)
        # tv.utils.save_image(img, img_name)

        i += 1

    logger.info('Generation done~~~')

