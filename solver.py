from common import *
from evaluation import *
from network import *


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.config = config

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model(config)

    def build_model(self, config):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=config.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=config.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=config.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(
                img_ch=3, output_ch=config.output_ch, t=self.t)
        elif self.model_type == 'Att_UNet':
            self.unet = Att_UNet(
                img_ch=3, output_ch=config.output_ch)
        else:
            raise NotImplementedError
        # # Load the pretrained Encoder
        # if os.path.isfile(config.pretrained):
        # 	self.unet.load_state_dict(torch.load(config.pretrained))
        # 	print('%s is Successfully Loaded from %s'%(self.model_type,config.pretrained))

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet = nn.DataParallel(self.unet)
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img*255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#



        # U-Net Train

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.

        unet_path = os.path.join(self.model_path, '%s-best.pkl' %(self.model_type))
        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0

            acc = 0.    # Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length = 0

            for i, data in enumerate(self.train_loader):
                # GT : Ground Truth
                images, GT = data[:2]
                images = images.to(self.device)
                GT = GT.to(self.device)

                SR = self.unet(images)

                if i == 0:
                    sample_in = images[0].detach().cpu().permute([
                        1, 2, 0]).numpy()
                    sample_in = (sample_in-sample_in.min()) / \
                        (sample_in.max()-sample_in.min())
                    sample_in *= 255
                    sample_in = sample_in.astype('uint8')
                    sample_in = np.concatenate([sample_in, sample_in], 1)
                    samples = SR.softmax(1).argmax(1)
                    sample_pred = samples[0].detach(
                    ).cpu().numpy().astype('uint8')
                    sample_gt = GT.argmax(1)[0].detach(
                    ).cpu().numpy().astype('uint8')
                    # import ipdb; ipdb.set_trace()
                    sample = np.concatenate([sample_pred, sample_gt], 1)
                    plt.figure()
                    plt.imshow(sample_in)
                    plt.imshow(sample, alpha=.5)
                    plt.savefig(f'cache/{epoch}_{i}.jpg')
                    # print('Save: ', f'cache/{epoch}_{i}.jpg')

                GT_flat = GT.permute([0, 2, 3, 1]).argmax(-1).view(-1)
                SR_flat = SR.permute(
                    [0, 2, 3, 1]).contiguous().view(-1, SR.size(1))

                loss = self.criterion(SR_flat, GT_flat)
                # loss = self.criterion(SR,GT)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                GT = GT.permute([0, 2, 3, 1]).contiguous().view(
                    [-1, GT.size(1)])
                aSR = SR.argmax(1).view(-1)
                eye = torch.eye(GT.size(1)).cuda()
                SR = eye[aSR, :]

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += 1

            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length

            # Print the log info
            print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                epoch+1, self.num_epochs,
                epoch_loss,
                acc, SE, SP, PC, F1, JS, DC))

            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

            #===================================== Validation ====================================#
            # self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length = 0
            for i, data in enumerate(self.valid_loader):
                images, GT = data[:2]
                images = images.to(self.device)
                GT = GT.to(self.device)
                with torch.no_grad():
                    SR = self.unet(images)
                if i == 0:
                    sample_in = images[0].detach().cpu().permute([
                        1, 2, 0]).numpy()
                    sample_in = (sample_in-sample_in.min()) / \
                        (sample_in.max()-sample_in.min())
                    sample_in *= 255
                    sample_in = sample_in.astype('uint8')
                    sample_in = np.concatenate([sample_in, sample_in], 1)
                    samples = SR.softmax(1).argmax(1)
                    sample_pred = samples[0].detach(
                    ).cpu().numpy().astype('uint8')
                    sample_gt = GT.argmax(1)[0].detach(
                    ).cpu().numpy().astype('uint8')
                    # import ipdb; ipdb.set_trace()
                    sample = np.concatenate([sample_pred, sample_gt], 1)
                    plt.figure()
                    plt.imshow(sample_in)
                    plt.imshow(sample, alpha=.5)
                    plt.savefig(f'sample_valid/{epoch}_{i}.jpg')
                    # print('Save: ', f'cache/{epoch}_{i}.jpg')

                GT = GT.permute([0, 2, 3, 1]).contiguous().view(
                    [-1, GT.size(1)])
                aSR = SR.argmax(1).view(-1)
                eye = torch.eye(GT.size(1)).cuda()
                SR = eye[aSR, :]

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)

                length += 1  # images.size(0)

            # import ipdb; ipdb.set_trace()
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC

            print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, , UNet-score: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, unet_score))

            # Save Best U-Net model

            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('Best %s model score : %.4f' %
                        (self.model_type, best_unet_score))
                torch.save(best_unet, unet_path)

    def test(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#

        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

        # U-Net Train
        # import ipdb; ipdb.set_trace()
        #===================================== Test ====================================#
        self.build_model(self.config)
        self.unet.load_state_dict(torch.load(self.config.pretrained))
        test_output_sample = self.config.pretrained.replace('.pkl', '')
        os.makedirs(test_output_sample, exist_ok=True)
        self.unet.train(True)
        # self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        length = 0
        for i, (images, GT, paths) in enumerate(self.test_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)
            with torch.no_grad():
                SR = self.unet(images).softmax(1)

            for j, (mask, img, path) in enumerate(zip(SR,images, paths)):
                name = os.path.basename(path)
                mask = mask.argmax(0).cpu().numpy()
                img = img.permute([1,2,0]).cpu().numpy()
                img = (img-img.min())/(img.max()-img.min())
                img = (img*255).astype('uint8')
                # output_image = 
                # output_image = np.stack([output_image]*3, axis=-1)
                # input_image = (images[0].cpu().permute(
                #     [1, 2, 0]).numpy()+1)*127.5
                output_path = os.path.join(test_output_sample, name)
                plt.figure()
                plt.imshow(img)
                plt.imshow(mask, alpha=.5)
                plt.savefig(output_path)
                plt.close()
                # # print(input_image.shape, output_image.shape)
                # combine = np.concatenate([input_image, output_image], axis=1)[
                #     :, :, ::-1]
                # cv2.imwrite(output_path, combine)
                # print('Output write at: ', output_path)
