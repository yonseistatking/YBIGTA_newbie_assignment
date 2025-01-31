import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


NUM_CLASSES = 10  

# 데이터 증강 설정
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

torch.cuda.empty_cache()

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")
    return total_loss / len(loader)

def main() -> None:
    # CIFAR-10 데이터셋 로드
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=data_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: ", device)

    # ResNet18 모델 선언 및 초기화
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()  # 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # AdamW 옵티마이저
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 학습률 감소 스케줄러

    # Early Stopping 변수 초기화
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 10

    # 학습 및 Early Stopping 루프
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train(model, train_loader, criterion, optimizer, device)
        current_loss = evaluate(model, test_loader, criterion, device)

        # Early Stopping 조건 체크
        if current_loss < best_loss:
            best_loss = current_loss
            early_stop_counter = 0  # 손실이 개선되었으므로 카운터 초기화
            torch.save(model.state_dict(), "best_model.pth")  # 베스트 모델 저장
        else:
            early_stop_counter += 1  # 손실이 개선되지 않으면 카운터 증가

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # 스케줄러 스텝 업데이트
        scheduler.step()

    print("Training finished.")
    print("Best model saved as 'best_model.pth'.")

    # 모델 저장
    torch.save(model.state_dict(), "resnet18_checkpoint.pth")
    print(f"Model saved to resnet18_checkpoint.pth")

if __name__ == "__main__":
    main()
