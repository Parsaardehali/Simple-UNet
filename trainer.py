from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Loss: {total_loss/len(dataloader):.4f}")