from re import L
import torch
from torch import nn
from repvgg import RepVGG
from time import perf_counter
import pandas as pd

torch.manual_seed(0)


def init(repvgg):
    for module in repvgg.modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    return repvgg


def benchmark(batches_sizes, device="cpu"):
    records = []
    # test the models
    with torch.no_grad():
        x = torch.randn((1, 3, 112, 112))
        repvgg = init(RepVGG([4, 8, 16], [1, 1, 1])).eval()
        out = repvgg(x)
        out_fast = repvgg.switch_to_fast()(x)
        assert torch.allclose(out, out_fast, atol=1e-5)

    print(f"{device=}")
    for batch_size in batches_sizes:
        x = torch.randn((batch_size, 3, 224, 224), device=torch.device(device))
        torch.cuda.reset_max_memory_allocated()
        with torch.no_grad():
            repvgg = (
                RepVGG([64, 128, 256, 512], [2, 2, 2, 2])
                .eval()
                .to(torch.device(device))
            )
            start = perf_counter()
            for _ in range(32):
                repvgg(x)

            records.append(
                {
                    "Type": "Default",
                    "VRAM (B)": torch.cuda.max_memory_allocated(),
                    "Time (s)": perf_counter() - start,
                    "batch size": batch_size,
                    "device": device,
                }
            )
            print(
                f"Memory without reparametrization {torch.cuda.max_memory_allocated()}"
            )
            print(f"Without reparametrization {perf_counter() - start:.2f}s")
            torch.cuda.reset_max_memory_allocated()
            repvgg.switch_to_fast().to(torch.device(device))
            start = perf_counter()
            for _ in range(32):
                repvgg(x)

            records.append(
                {
                    "Type": "Fast",
                    "VRAM (B)": torch.cuda.max_memory_allocated(),
                    "Time (s)": perf_counter() - start,
                    "batch size": batch_size,
                    "device": device,
                }
            )
            print(f"With reparametrization {perf_counter() - start:.2f}s")
            print(f"Memory with reparametrization {torch.cuda.max_memory_allocated()}")

    return pd.DataFrame.from_records(records)


# df = benchmark([1, 2, 4, 8, 16, 32, 64, 128], "cuda")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.style.use("science")

    batches_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    df = benchmark(batches_sizes, "cuda")
    print(df)

    fig = plt.figure()

    default_time = df[df.loc[:, "Type"] == "Default"].loc[:, "Time (s)"]
    fast_time = df[df.loc[:, "Type"] == "Fast"].loc[:, "Time (s)"]

    plt.plot(batches_sizes, default_time.values, label="default")
    plt.plot(batches_sizes, fast_time.values, label="reparametrization")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("time.png", dpi=800)

    fig = plt.figure()

    default_time = df[df.loc[:, "Type"] == "Default"].loc[:, "VRAM (B)"]
    fast_time = df[df.loc[:, "Type"] == "Fast"].loc[:, "VRAM (B)"]

    plt.plot(batches_sizes, default_time.values, label="default")
    plt.plot(batches_sizes, fast_time.values, label="reparametrization")

    plt.xlabel("Batch Size")
    plt.ylabel("VRAM (B)")
    plt.legend()
    plt.savefig("vram.png", dpi=800)
