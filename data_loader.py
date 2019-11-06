from common import *


def imread(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.img_paths = glob(os.path.join(root, '*.png'))
        self.imgs = [imread(path) for path in self.img_paths]
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.imgs)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        img_gt = self.imgs[index]
        h, w = img_gt.shape[:2]
        image = img_gt[:, :w//2]
        image = Image.fromarray(image)
        gt = img_gt[:, w//2:]
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        GT = Image.fromarray(gt)

        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(
            T.Resize((int(ResizeRange*aspect_ratio), ResizeRange)))
        p_transform = random.random()

        if self.mode == 'train':
            RotationRange = random.randint(-180, 180)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio

            Transform.append(T.RandomRotation(
                (RotationDegree, RotationDegree)))

            CropRange = random.randint(250, 270)
            Transform.append(T.CenterCrop(
                (int(CropRange*aspect_ratio), CropRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(
                box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper,
                              ShiftRange_right, ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = Transform(image)

            Transform = []

        Transform.append(T.Resize((256, 256)))
        # Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        # print(image.shape)
        return image, GT, self.img_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.imgs)


class ImageFolderOCRSegment(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4, num_ch=14):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.num_ch = num_ch
        self.mode = mode

        self.img_paths = list(sorted(glob(os.path.join(root, 'A', '*.png'))))
        self.label_paths = list(sorted(glob(os.path.join(root, 'B', '*.png'))))
        train_len = int(.8*len(self.img_paths))
        if mode == 'train':
            self.img_paths = self.img_paths[:train_len]
            self.label_paths = self.label_paths[:train_len]
        else:
            self.img_paths = self.img_paths[train_len:]
            self.label_paths = self.label_paths[train_len:]

        self.imgs = [imread(path) for path in self.img_paths]
        self.lbls = [cv2.imread(path, 0) for path in self.label_paths]

        self.image_size = image_size
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

        data = self.__getitem__(0)
        # import ipdb; ipdb.set_trace()
        print("image count in {} path :{}".format(self.mode, len(self.imgs)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        # img_gt = self.imgs[index]
        # image = img_gt[:,:w//2]
        image = self.imgs[index]
        # h, w = img_gt.shape[:2]
        image = Image.fromarray(image)
        # gt = img_gt[:,w//2:]
        gt = self.lbls[index]
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        # import ipdb; ipdb.set_trace()
        GT = []
        eye = np.eye(self.num_ch)
        gt = eye[gt]

        # _gt = gt == 0
        # _gt = (_gt*255).astype('uint8')*0
        # GT.append(Image.fromarray(_gt))
        for i in range(self.num_ch):
            _gt = gt[:,:,i]
            if i == 0:
                _gt = 1-_gt

            _gt = (_gt*255).astype('uint8')
            GT.append(Image.fromarray(_gt))

        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(
            T.Resize((int(ResizeRange*aspect_ratio), ResizeRange)))
        p_transform = random.random()

        if self.mode == 'train':
            RotationRange = random.randint(-180, 180)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))

        # if (self.mode == 'train') and p_transform <= self.augmentation_prob:
        #     RotationDegree = random.randint(0, 3)
        #     RotationDegree = self.RotationDegree[RotationDegree]
        #     if (RotationDegree == 90) or (RotationDegree == 270):
        #         aspect_ratio = 1/aspect_ratio

        #     Transform.append(T.RandomRotation(
        #         (RotationDegree, RotationDegree)))

        #     CropRange = random.randint(250, 270)
        #     Transform.append(T.CenterCrop(
        #         (int(CropRange*aspect_ratio), CropRange)))
        #     Transform = T.Compose(Transform)

        #     image = Transform(image)
        #     GT = [Transform(_GT) for _GT in GT]

        #     ShiftRange_left = random.randint(0, 20)
        #     ShiftRange_upper = random.randint(0, 20)
        #     ShiftRange_right = image.size[0] - random.randint(0, 20)
        #     ShiftRange_lower = image.size[1] - random.randint(0, 20)
        #     image = image.crop(
        #         box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
        #     GT = [_GT.crop(box=(ShiftRange_left, ShiftRange_upper,
        #                         ShiftRange_right, ShiftRange_lower)) for _GT in GT]

        #     if random.random() < 0.5:
        #         image = F.hflip(image)
        #         GT = [F.hflip(_GT) for _GT in GT]

        #     if random.random() < 0.5:
        #         image = F.vflip(image)
        #         GT = [F.vflip(_GT) for _GT in GT]

        #     Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

        #     image = Transform(image)

        #     Transform = []

        Transform.append(T.Resize((256, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        # import ipdb; ipdb.set_trace()
        GT_out = []
        for i in range(self.num_ch):
            x = Transform(GT[i]) 
            if i == 0:
                x = 1-x
            GT_out.append(x)
        GT =GT_out
        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        return image, (torch.cat(GT) > .5).float(), self.img_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.imgs)


# def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
# 	"""Builds and returns Dataloader."""

# 	if mode == 'test':
# 		dataset_test = ImageFolder(root = image_path, image_size =image_size, mode='test',augmentation_prob=-1)
# 		image_path = image_path.replace('/test', '/train')
# 		dataset_train = ImageFolder(root = image_path, image_size =image_size, mode='test',augmentation_prob=-1)

# 		dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
# 	else:
# 		dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
# 	data_loader = data.DataLoader(dataset=dataset,
# 								  batch_size=batch_size if mode=='train' else 1,
# 								  shuffle=True,
# 								  num_workers=num_workers)
# 	return data_loader


def get_loader_ocr(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4, num_ch=14):
    """Builds and returns Dataloader."""

    if mode == 'test':
        dataset_test = ImageFolderOCRSegment(
            root=image_path, image_size=image_size, mode='test', augmentation_prob=-1, num_ch=num_ch)
        image_path = image_path.replace('/test', '/train')
        dataset_train = ImageFolderOCRSegment(
            root=image_path, image_size=image_size, mode='test', augmentation_prob=-1, num_ch=num_ch)
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    else:
        dataset = ImageFolderOCRSegment(
            root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob, num_ch=num_ch)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,# if mode == 'train' else 1
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


def onehot_to_heatmap(onehot, save_name='test'):
    if isinstance(onehot, torch.Tensor):
        onehot = onehot.cpu().numpy()

    os.makedirs('cache', exist_ok=1)
    b = np.argmax(onehot, 0).numpy().astype('uint8')
    plt.figure()
    plt.imshow(b)
    plt.savefig(f'cache/{save_name}.png')
    plt.close()
