from torchvision.datasets import Food101

train_dataset = Food101(
    root="./data",
    split="train",
    download=True
)

test_dataset = Food101(
    root="./data",
    split="test",
    download=True
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Classes:", len(train_dataset.classes))
print(train_dataset.classes[:10])