
import torch
import models.networks as networks
import util.util as util


class StageII_MultiAtt3_Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_ganFeat_loss:
                self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if not opt.no_parsing_loss:
                self.criterionParsing = networks.ParsingCrossEntropyLoss(tensor=self.FloatTensor)
            if not opt.no_style_loss:
                self.criterionStyle = networks.StyleLoss()
            if not opt.no_tv_loss:
                self.criterionTV = networks.TVLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        inputs, real_img, face_mask, part_mask, mask_onehot = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(inputs, real_img, face_mask, part_mask, mask_onehot)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(inputs, real_img)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image = self.netG(inputs)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain and not opt.no_GAN:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = None
        if not self.opt.no_GAN:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if not self.opt.no_GAN:
            util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################
    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = None
        if not opt.no_GAN:
            netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and not opt.no_GAN:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    def lable_2_onehot(self, label):
        bs, _, h, w = label.size()
        input_label = self.FloatTensor(bs, self.opt.label_nc, h, w).zero_()
        onehot_label = input_label.scatter_(1, label.long(), 1.0)

        return onehot_label

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            original_image = data['original_image'].cuda()
            data['incompleted_image'] = data['incompleted_image'].cuda()
            data['mask_edge'] = data['mask_edge'].cuda()
            data['mask_noise'] = data['mask_noise'].cuda()
            data['mask_color'] = data['mask_color'].cuda()
            data['part_mask'] = data['part_mask'].cuda()

            data['face_mask'] = data['face_mask'].cuda()
            data['mask_onehot'] = data['mask_onehot'].cuda()



        if not self.opt.no_parsing_input:
            # create one-hot label map
            original_label = data['label'].cuda()
            original_label = self.lable_2_onehot(original_label)
            inputs = torch.cat([original_label, data['incompleted_image'], data['part_mask'], data['mask_noise'], data['mask_color'], data['mask_edge']], dim=1)
            # print("inputs.shape")  torch.Size([1, 29, 512, 320])
            # print(inputs.shape)
        else:
            # ablation no parsing,
            inputs = torch.cat([data['incompleted_image'], data['mask_onehot'], data['mask_noise'], data['mask_color'], data['mask_edge']], dim=1)

        return inputs, original_image, data['face_mask'], data['part_mask'], data['mask_onehot']

    def compute_generator_loss(self, inputs, real_image, face_mask, part_mask, mask_onehot):
        G_losses = {}

        fake_image = self.netG(inputs)

        if not self.opt.no_GAN:
            pred_fake, pred_real = self.discriminate(inputs, fake_image, real_image)
            G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D

                G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        if not self.opt.no_L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * self.opt.lambda_L1

        if not self.opt.no_parsing_loss:
            G_losses['Parsing'] = self.criterionParsing(fake_image, real_image) * self.opt.lambda_parsing

        if not self.opt.no_face_loss:
            G_losses['Face'] = self.criterionL1(fake_image * face_mask, real_image * face_mask) * self.opt.lambda_face

        if not self.opt.no_mask_loss:
            G_losses['Mask'] = self.criterionL1(fake_image * mask_onehot, real_image * mask_onehot) * self.opt.lambda_mask

        if not self.opt.no_foreground_loss:
            G_losses['Foreground'] = self.criterionL1(fake_image * part_mask, real_image * part_mask) * self.opt.lambda_foreground

        if not self.opt.no_style_loss:
            G_losses['Style'] = self.criterionStyle(fake_image * mask_onehot, real_image * mask_onehot) * self.opt.lambda_style

        if not self.opt.no_tv_loss:
            real_face_image = real_image * face_mask
            fake_face_image = fake_image * face_mask
            fake_irr_mask = face_mask * mask_onehot
            let_fake_irr_mask = face_mask - fake_irr_mask

            tv_image = real_face_image * let_fake_irr_mask + fake_face_image * fake_irr_mask
            G_losses['Face_TV'] = self.criterionTV(tv_image) * self.opt.lambda_tv

        return G_losses, fake_image

    def compute_discriminator_loss(self, inputs, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.netG(inputs)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(inputs, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    def discriminate(self, inputs, fake_image, real_image):
        fake_concat = torch.cat([inputs, fake_image], dim=1)
        real_concat = torch.cat([inputs, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real


    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
