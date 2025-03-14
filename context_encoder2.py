import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64 , kernel_size=4, stride=2, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)  
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) 
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) 
        self.relu5 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x)) 
        x = self.relu2(self.conv2(x)) 
        x = self.relu3(self.conv3(x))  
        x = self.relu4(self.conv4(x))  
        x = self.relu5(self.conv5(x)) 

        return x

class FullyConnected(nn.Module):
    def __init__(self, batch_size=128):
        super(FullyConnected, self).__init__()
        self.batch_size = batch_size

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*4*4, 512*4*4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.fc(x))
        output = x = x.view(-1, 512, 4, 4)
        
        return output
    
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.upconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu1(self.upconv1(x))
        x = self.relu2(self.upconv2(x))
        x = self.relu3(self.upconv3(x))
        x = self.relu4(self.upconv4(x))
        x = self.relu5(self.upconv5(x))

        return x
    
class ContextEncoder2(nn.Module):
    def __init__(self, batch_size):
        super(ContextEncoder2, self).__init__()

        self.batch_size = batch_size

        self.encode = Encoder2()
        if (self.batch_size != 128):
            self.fc = FullyConnected(self.batch_size)
        else:
            self.fc = FullyConnected()
        self.decode = Decoder2()
    
    def forward(self, x):
        encoded = self.encode(x)
        connection = self.fc(encoded)
        decoded = self.decode(connection)

        return decoded

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#########################################
# Example usage
#########################################
if __name__ == "__main__":
    # Create a model instance
    model = ContextEncoder2()
    input_tensor = torch.randn(128, 3, 128, 128)
    output = model(input_tensor)
    print("Output shape:", output.shape)




        

        