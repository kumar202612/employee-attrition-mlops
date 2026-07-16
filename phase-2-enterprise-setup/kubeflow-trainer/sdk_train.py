from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.options.common import Name


NAMESPACE  = "default"
NUM_NODES  = 2
RUNTIME    = "torch-distributed" 
JOB_NAME   = "distributed-training"


def train_func():

    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}/{world_size}] Starting...")
    dist.init_process_group(backend="gloo")

    torch.manual_seed(42)

    num_samples  = 1000
    num_features = 20
    num_classes  = 5

    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset    = TensorDataset(X, y)
    sampler    = DistributedSampler(dataset, num_replicas=world_size,
                                    rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    print(f"[Rank {rank}] Dataset: {num_samples} samples | "
          f"Batches per epoch: {len(dataloader)}")

    model = nn.Sequential(
        nn.Linear(num_features, 64), nn.ReLU(),
        nn.Linear(64, 32),           nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    model     = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        model.train()
        total_loss, correct, total = 0, 0, 0

        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred        = output.argmax(dim=1)
            correct    += pred.eq(target).sum().item()
            total      += target.size(0)

        print(f"[Rank {rank}] Epoch {epoch+1}/5 | "
              f"Loss: {total_loss/len(dataloader):.4f} | "
              f"Accuracy: {100.*correct/total:.2f}%")

    dist.barrier()
    if rank == 0:
        torch.save(model.module.state_dict(), "/tmp/model.pt")
        print("[Rank 0] Model saved to /tmp/model.pt")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done!")


def main():
    client = TrainerClient(backend_config=KubernetesBackendConfig(namespace=NAMESPACE))

    print(f"Submitting TrainJob")
    print(f"  Name     : {JOB_NAME}")
    print(f"  Runtime  : {RUNTIME}")
    print(f"  Nodes    : {NUM_NODES}")
    print(f"  Namespace: {NAMESPACE}\n")

    job_name = client.train(
        runtime=RUNTIME,
        options=[Name(JOB_NAME)],
        trainer=CustomTrainer(
            func=train_func,
            num_nodes=NUM_NODES,
            resources_per_node={
                "cpu":    "1",
                "memory": "2Gi",
            },
        ),
    )
    print(f"TrainJob created: {job_name}\n")

    print("Waiting for job to complete...")
    client.wait_for_job_status(name=job_name, timeout=600)

    print("\nLogs from node-0:")
    for line in client.get_job_logs(name=job_name, step="node-0", follow=True):
        print(line)

    print("\nLogs from node-1:")
    for line in client.get_job_logs(name=job_name, step="node-1", follow=True):
        print(line)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
