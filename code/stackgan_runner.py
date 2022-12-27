from itertools import product
from multiprocessing import Pool, cpu_count, Process
from utils import mkdir_p, save_model, save_img_results, KL_loss, compute_discriminator_loss, compute_generator_loss, weights_init
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
from PIL import Image
import torchfile
from torchvision import transforms
import tensorflow as tf

from tensorboardX import FileWriter
from configuration import cfg


    
def cal_G_loss(netD, fake_imgs, real_labels, cond):
    criterion = nn.BCELoss()
    cond = cond.detach()
    fake_f = netD(fake_imgs)

    fake_cond_ouput = netD.get_cond_logits(fake_f, cond)
    errD_fake = criterion(fake_cond_ouput, real_labels)
    if netD.get_uncond_logits is not None:
        fake_uncond_output = netD.get_uncond_logits(fake_f)
        uncond_errD_fake = criterion(fake_uncond_output, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake

def cal_d_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, cond):
        criterion = nn.BCELoss()
        batch_size = real_imgs.size(0)
        cond = cond.detach()
        fake = fake_imgs.detach()

        real_img_feature = netD(real_imgs)
        fake_img_feature = netD(fake)

        real_output = netD.get_cond_logits(real_img_feature, cond)
        errD_real  = criterion(real_output, real_labels)
        wrong_output = netD.get_cond_logits(real_img_feature[:(batch_size-1)], cond[1:])
        errD_wrong = criterion(wrong_output, fake_labels[1:])

        fake_output = netD.get_cond_logits(fake_img_feature, cond)
        errD_fake= criterion(fake_output, fake_labels)

        if netD.get_uncond_logits is not None:
            real_uncond_output = netD.get_uncond_logits(real_img_feature)
            errD_real_uncond = criterion(real_uncond_output, real_labels)

            fake_uncond_output = netD.get_uncond_logits(fake_img_feature)
            errD_fake_uncond = criterion(fake_uncond_output, fake_labels)

            errD = (errD_real+errD_real_uncond)/2. + (errD_fake+errD_wrong+errD_fake_uncond)/3.
            errD_real =  (errD_real+errD_real_uncond)/2
            errD_fake = (errD_fake+errD_fake_uncond)/2.
        else:
            errD = errD_real + (errD_fake+errD_wrong)*0.5
        return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()

class GAN(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        
        self.batch_size = cfg.TRAIN.BATCH_SIZE 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cudnn.benchmark = True


    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        netD = STAGE1_D()
        netD.apply(weights_init)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=torch.device('cpu'))
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=torch.device('cpu'))
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=torch.device('cpu'))
            netG.load_state_dict(state_dict,strict=False)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=torch.device('cpu'))
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict, strict=False)
            print('Load from: ', cfg.NET_D)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD
    
    

    def train(self, data_loader, stage=1):
       
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        with torch.no_grad():
            fixed_noise = torch.FloatTensor(batch_size,nz).normal_(0,1)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            print(start_t)
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
        
            for i, data in enumerate(data_loader, 0):
               
                # Prepare training data
                real_img_cpu, txt_embedding = data
                real_imgs = real_img_cpu
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()
                
                
                
                # Generate fake images
                noise.data.normal_(0, 1)
                inputs = [txt_embedding, noise]
                print(txt_embedding.shape)
                fake_imgs, mu, logvar =netG(txt_embedding, noise)

                #print(inputs[0])
                #
                # print(inputs[1])
            
                
                
                

                # Update D network
               
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = cal_d_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, mu)
                errD.backward()
                
                optimizerD.step()
                
                # Update G network
                
                netG.zero_grad()
                errG = cal_G_loss(netD, fake_imgs,real_labels, mu)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1
                if i % 100 == 0:
                    # save the image result for each epoch
                    #inputs = [txt_embedding, fixed_noise]
                    fake, _, _ = netG(txt_embedding, fixed_noise)
                    # lr_fake, fake, _, _ = \
                    #     pooling.map(netG, inputs)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(), kl_loss.item(),
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        
        save_model(netG, netD, self.max_epoch, self.model_dir)

    



    def sample(self,text_embd, token_size ,stage=2):

        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()


        # Load text embeddings generated from the encoder
        embeddings = text_embd
        num_embeddings = 1
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            print(txt_embedding.shape)
            print(noise.shape)
            stage_1, fake_imgs, mu, logvar = \
                netG(txt_embedding,noise)
            for i in range(batch_size):
                print(batch_size)
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

