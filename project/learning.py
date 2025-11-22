import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def get_trainde_model(model, train_loader: DataLoader, EPOCHS=100, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda') # Для ускорения FP16

    print("\nНачинаем обучение Transformer...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Autocast делает магию смешанной точности (быстрее на 3080 Ti)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backprop с масштабированием градиентов
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

    return model