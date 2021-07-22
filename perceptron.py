from torch import nn


'''
    Perceptron
'''
class Perceptron(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''Forward pass'''
        return self.relu(self.fc(x))

'''
if __name__ == '__main__':

    # Hyperparameters
    batch_size = 10
    learning_rate = 1e-4
    epochs = 5

    # data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

    # dataset
    dataset = CIFAR10(root='./data/cifar10',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Initializing Net
    net = Perceptron()

    # defining loss function
    loss_function = nn.CrossEntropyLoss()
    # defining optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)


    # training loop
    for epoch in range(epochs):
        
        print('Epoch %s' % (epoch+1))

        current_loss = 0.0

        # iterate over the data
        for i, data in enumerate(trainloader, 0):
            
            # get data and ground truth
            inputs, targets = data

            # set gradients of all optimized tensors to zero
            optimizer.zero_grad()

            # forward pass of data through net
            outputs = net(inputs)

            # compute loss
            loss = loss_function(outputs, targets)

            # backward pass
            loss.backward()

            # optimizing parameters
            optimizer.step()

            # show stats
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f'% (i+1, current_loss/500))
                current_loss = 0.0

    # end
    print('Training finished!')'''
