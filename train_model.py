def train_model(model, optimizer, train_loader,num_epochs=3 ):
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss.item()}")