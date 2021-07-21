import torchvision, torch, random
from torchvision import transforms
import matplotlib.pyplot as plt

class dataholder:
    def loader(self, download=False, num=None):
        train_transform = self.transform_train()
        test_transform = self.transform_test()
        trainset = torchvision.datasets.CIFAR10(root='../data/', train=True,download=download, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='../data/', train=False,download=False, transform=test_transform)

        cifar_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset.transform, testset.transform = cifar_process, cifar_process
        trainset.transforms, testset.transforms = torchvision.datasets.vision.StandardTransform(cifar_process), torchvision.datasets.vision.StandardTransform(cifar_process)
        
        trainx, trainy = torch.tensor(trainset.data).permute(0,3,1,2), torch.tensor(trainset.targets)
        testx, testy = torch.tensor(testset.data).permute(0,3,1,2), torch.tensor(testset.targets)
        if num:
            trainx, trainy = trainx[:num], trainy[:num]
        print(trainx.shape, trainy.shape, testx.shape, testy.shape)
        self.trainx, self.trainy, self.testx, self.testy = trainx, trainy, testx, testy
        self.classes = trainset.classes
        self.train = trainset
        self.test = testset
        print(trainset.__dict__.keys())
    def transform_train(self, ):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    def transform_test(self, ):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    def visualize(self, ):
        x, y = self.trainx, self.trainy
        num_img_per_class = 12
        classes = self.classes
        samples = []
        for i, cls in enumerate(classes):
            plt.text(-4, 34 * i + 18, cls, ha='right')
            idxs = (y == i).nonzero(as_tuple=True)[0]
            for j in range(num_img_per_class):
                img = x[idxs[random.randrange(idxs.shape[0])]]
                samples.append(img)
        img_grid = torchvision.utils.make_grid(samples, nrow=num_img_per_class, padding=2)
        print(type(img_grid), img_grid.shape)
        plt.axis('off')
        plt.imshow(img_grid.permute(1,2,0))
        plt.savefig("sample.png")

