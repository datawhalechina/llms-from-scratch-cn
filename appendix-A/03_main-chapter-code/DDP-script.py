# 附录A ：PyTorch的介绍（第三部分）

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 导入新的库
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 创建一个新的函数用于初始化一个分布式进程（每个GPU一个进程）
# 该函数允许进程之间的通信
def ddp_setup(rank, world_size):
    """
    提示：
        rank:特定的进程编号（进程ID)
        world_size:组内的进程总数
    """
    
    # 正在运行的机器编号 ID：进程0
    # 这里的前提是假设所有的GPU在同一台机器上
    os.environ["MASTER_ADDR"] = "localhost"
    # 机器上任意的空闲端口号
    os.environ["MASTER_PORT"] = "12345"

    # 初始化进程
    # Windows 用户使用"gloo"来替代下面代码中的"nccl"
    # nccl: NVIDIA Collective Communication Library
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False, # 这里设置为False 
        pin_memory=True,
        drop_last=True,
        # 在多个GPU上划分批次，确保批次之间不重叠样本
        sampler=DistributedSampler(train_ds) 
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# 包装器
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size) # 

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank]) # 使用分布式数据并行（DDP）将模型进行包装
    # 现在核心模型可以通过 model.module 访问
    
    for epoch in range(num_epochs):
    
        model.train()
        for features, labels in enumerate(train_loader):
    
            features, labels = features.to(rank), labels.to(rank) 
            logits = model(features)
            loss = F.cross_entropy(logits, labels) # 损失函数
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            ### 日志
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
    
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group() # 清理退出分布式模式


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)

    # 新建进程
    # 请注意，spawn会自动传递排名
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size 会为每个GPU生成一个进程

