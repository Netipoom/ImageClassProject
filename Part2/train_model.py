import torch
from time import time
from loadData import loadData
from workspace_utils import active_session
def train_model(model,criterion,optimizer,epochs,data_dir,gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")
    model.to(device)
    
    trainloaders = loadData(data_dir)['train']
    validloaders = loadData(data_dir)['valid']
    steps = 0
    running_loss = 0
    print_every = 5
    with active_session():
        for epoch in range(int(epochs)):
            start = time()
            for inputs, labels in trainloaders:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Valid loss: {test_loss/len(validloaders):.3f}.. "
                        f"Valid accuracy: {accuracy/len(validloaders):.3f}")
                    running_loss = 0
                    model.train()

            end_time=time()
            tot_time=end_time-start
            print(f"Epoch {epoch+1}..total time: {tot_time}")
    return model,optimizer