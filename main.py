import data, model 
import torch
import os, time
from torch.utils.tensorboard import SummaryWriter

time.tzset()

def train(model_name, optim_name, epoch_continue=None):
    dataCls = data.dataholder()
    dataCls.loader(num=100)
    dataCls.visualize()
    trainloader = torch.utils.data.DataLoader(dataCls.train, batch_size=30, shuffle=True, num_workers=2, )

    net = model.resnet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    if epoch_continue:
        loaded_model = torch.load(os.path.join(os.getcwd(), "model", model_name+"_"+str(epoch_continue)))
        net.load_state_dict(loaded_model)
        net.eval() # Some layers like Dropout or Batchnorm won’t work properly if you don’t call net.eval() after loading.
        loaded_optim = torch.load(os.path.join(os.getcwd(), "model", optim_name+"_"+str(epoch_continue)))
        optimizer.load_state_dict(loaded_optim)
    else:
        epoch_continue = 0
    
    writer = SummaryWriter()

    running_loss = 0
    print_freq = 1e2
    epochs = 10 + epoch_continue
    for epoch in range(epoch_continue, epochs):
        for i, data_batch in enumerate(trainloader):
            x, y = data_batch
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % print_freq == 0:
                print(epoch, i+1, running_loss / print_freq, time.strftime('%X %x %Z'))
                running_loss = 0

    print("----- Weights: -----")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    print("----- Optimizer: -----")
    print(optimizer.state_dict().keys())
    print(optimizer.state_dict()['param_groups'])

    
    torch.save(net.state_dict(), os.path.join(os.getcwd(), "model", model_name+"_"+str(epochs)))
    torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), "model", optim_name+"_"+str(epochs)))

    writer.flush()
    writer.close()

def test(model_name, epoch_continue=1):
    dataCls = data.dataholder()
    dataCls.loader(num=100)
    
    net = model.resnet()
    testloader = torch.utils.data.DataLoader(dataCls.test, batch_size=30, shuffle=True, num_workers=2, )
    
    loaded_model = torch.load(os.path.join(os.getcwd(), "model", model_name+"_"+str(epoch_continue)))
    net.load_state_dict(loaded_model)

    net.eval() # Some layers like Dropout or Batchnorm won’t work properly if you don’t call net.eval() after loading.

    # test
    total = 0
    correct = 0
    total_per_class = torch.zeros(10)
    correct_per_class = torch.zeros(10)
    with torch.no_grad():
        for i, test_batch in enumerate(testloader):
            x, y = test_batch
            out = net(x)
            _, pred = torch.max(out.data, 1)
            total += y.size()[0]
            correct += (y == pred).sum().item()
            total_per_class += torch.histc(y.float(), bins=10, min=0, max=9)
            correct_per_class += torch.histc(y[y == pred].float(), bins=10, min=0, max=9)
    print("Total accuracy is {}".format(correct / total))
    print("Accuracy per class is {}".format(correct_per_class / total_per_class))

if __name__ == '__main__':
    print("Working dir is " + os.getcwd())
    model_name = "resnet_model.pth"
    optim_name = "resnet_optim.pth"
    train(model_name, optim_name, epoch_continue=24)
    test(model_name, epoch_continue=24)

