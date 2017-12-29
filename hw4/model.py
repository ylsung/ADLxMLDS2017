import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc, ntext, dim, image_size):
        super(Generator, self).__init__()
        assert image_size % 16 == 0, 'image_size should be multiple of 16'

        t_dim, t_image_size = dim // 2, 4
        while t_image_size != image_size:
            t_dim = t_dim * 2
            t_image_size = t_image_size * 2

        self.module_list = []

        prepro_layer = nn.Sequential(
            nn.ConvTranspose2d(nz+ntext, t_dim, 4, 1, 0),
            nn.BatchNorm2d(t_dim),
            nn.ReLU(True),
            )

        self.module_list.append(prepro_layer)

        t_image_size = 4
        while t_image_size < image_size // 2:

            conv_layer = nn.Sequential(
                nn.ConvTranspose2d(t_dim, t_dim//2, 4, 2, 1),
                nn.BatchNorm2d(t_dim//2),
                nn.ReLU(True),
                )
            self.module_list.append(conv_layer)
            t_dim = t_dim // 2
            t_image_size = t_image_size * 2

        output_layer = nn.Sequential(
            nn.ConvTranspose2d(t_dim, nc, 4, 2, 1),
            nn.Sigmoid(),
            )

        self.module_list.append(output_layer)

        self.module_list = nn.ModuleList(self.module_list)
    def forward(self, inputs, tags):
        tags = tags.view(tags.size(0), tags.size(1), 1, 1)
        output = torch.cat((inputs, tags), dim=1)
        for i, layer in enumerate(self.module_list):
            output = layer(output)
        return output


class Critic(nn.Module):
    def __init__(self, nz, nc, ntext, dim, image_size):
        super(Critic, self).__init__()

        self.module_list = []

        prepro_layer = nn.Sequential(
            nn.Conv2d(nc, dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.module_list.append(prepro_layer)
        image_size //= 2


        while image_size > 4:
            conv_layer = nn.Sequential(
                nn.Conv2d(dim, dim*2, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                )

            self.module_list.append(conv_layer)

            image_size //= 2
            dim *= 2

        # self.output_layer = nn.Conv2d(dim+ntext, 1, 4, 1, 0)
        self.binary_classifier = nn.Conv2d(dim, 1, 4, 1, 0)
        self.style_classifier = nn.Sequential(
            nn.Conv2d(dim, ntext, 4, 1, 0),
            nn.Sigmoid(),
            )
            
        
        # self.module_list.append(output_layer)
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, inputs, tags):
        for i, layer in enumerate(self.module_list):
            output = layer(inputs if i == 0 else output)
        # tags = tags.view(tags.size(0), tags.size(1), 1, 1)
        # tags = tags.expand(tags.size(0), tags.size(1), 4, 4)
        # output = torch.cat((output, tags), dim=1)
        # return self.output_layer(output)
        binary = self.binary_classifier(output).view(output.size(0), -1)
        style = self.style_classifier(output).view(output.size(0), -1)
        return binary, style

class Generator_img2img(nn.Module):
    def __init__(self, nz, nc, ntext, dim, image_size):
        super(Generator_img2img, self).__init__()
        assert image_size % 16 == 0, 'image_size should be multiple of 16'

        # input should be a image

        # downsampling

        downsampling = [
            nn.Conv2d(nz, dim, 4, 2, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, 4, 2, 0),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.Conv2d(dim*2, dim*8, 4, 2, 0), # dim=256
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            ]
        upsampling = [
            nn.ConvTranspose2d(dim*8+ntext, dim*4, 4, 2, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim*2, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, nc, 4, 2, 1),
            nn.Sigmoid(),
            ]

        self.downsampling = nn.Sequential(*downsampling)
        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, inputs, tags):
        tags = tags.view(tags.size(0), tags.size(1), 1, 1)
        tags = tags.expand(tags.size(0), tags.size(1), 6, 6)
        output = self.downsampling(inputs)
        output = torch.cat((output, tags), dim=1)
        return self.upsampling(output)

class Critic_img2img(nn.Module):
    def __init__(self, nz, nc, ntext, dim, image_size):
        super(Critic_img2img, self).__init__()

        self.module_list = []

        prepro_layer = nn.Sequential(
            nn.Conv2d(nc, dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.module_list.append(prepro_layer)
        image_size //= 2


        while image_size > 4:
            conv_layer = nn.Sequential(
                nn.Conv2d(dim, dim*2, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                )

            self.module_list.append(conv_layer)

            image_size //= 2
            dim *= 2

        # self.output_layer = nn.Conv2d(dim+ntext, 1, 4, 1, 0)
        self.binary_classifier = nn.Conv2d(dim, 1, 3, 1, 0)
        self.style_classifier = nn.Sequential(
            nn.Conv2d(dim, ntext, 3, 1, 0),
            nn.Sigmoid(),
            )
            
        
        # self.module_list.append(output_layer)
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, inputs, tags):
        for i, layer in enumerate(self.module_list):
            output = layer(inputs if i == 0 else output)
        # tags = tags.view(tags.size(0), tags.size(1), 1, 1)
        # tags = tags.expand(tags.size(0), tags.size(1), 4, 4)
        # output = torch.cat((output, tags), dim=1)
        # return self.output_layer(output)
        binary = self.binary_classifier(output).view(output.size(0), -1)
        style = self.style_classifier(output).view(output.size(0), -1)
        return binary, style

if __name__ == '__main__':
    C = Critic_img2img(nz=100, nc=3, ntext=25, dim=64, image_size=96)
    print(C)
    input_ = torch.rand(28, 3, 96, 96)
    tags = torch.rand(28, 23, 1, 1)
    input_ = torch.autograd.Variable(input_)
    tags = torch.autograd.Variable(tags)
    print(C(input_, tags))

    # G = Generator(nz=100, nc=3, ntext=25, dim=64, image_size=64)

    # print(G)
    # input_ = torch.rand(28, 100, 1, 1)
    # tags = torch.rand(28, 25, 1, 1)
    # input_ = torch.autograd.Variable(input_)
    # tags = torch.autograd.Variable(tags)
    # print(G(input_, tags))

    # G = Generator_img2img(nz=3, nc=3, ntext=25, dim=64, image_size=64)

    # print(G)
    # input_ = torch.rand(28, 3, 64, 64)
    # tags = torch.rand(28, 25, 1, 1)
    # input_ = torch.autograd.Variable(input_)
    # tags = torch.autograd.Variable(tags)
    # print(G(input_, tags))