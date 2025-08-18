import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple


def benchmark_function(
    func: Callable, tensor: torch.Tensor, device: str, num_runs: int = 100
) -> float:
    """Benchmark a function on a specific device."""
    tensor = tensor.to(device)

    # Warm up
    for _ in range(5):
        result = func(tensor)
        if device == "mps":
            torch.mps.synchronize()

    # Actual timing
    start_time = time.time()
    for _ in range(num_runs):
        result = func(tensor)
        if device == "mps":
            torch.mps.synchronize()  # Ensure GPU operations complete
    end_time = time.time()

    return (end_time - start_time) / num_runs


def test_operations():
    """Test various operations at different tensor sizes."""

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS not available! Make sure you have PyTorch with MPS support.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("-" * 60)

    # Define test operations
    operations = {
        "Element-wise multiply": lambda x: x * 2.5,
        "Element-wise add": lambda x: x + 1.0,
        "Square": lambda x: x**2,
        "Sine": lambda x: torch.sin(x),
        "Sum reduction": lambda x: torch.sum(x),
        "Matrix multiply (square)": lambda x: torch.mm(x, x.T)
        if x.dim() == 2
        else torch.matmul(x, x.transpose(-2, -1)),
        "Softmax": lambda x: torch.softmax(x, dim=-1),
        "ReLU": lambda x: torch.relu(x),
    }

    # Define tensor sizes to test (number of elements)
    sizes = [
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
    ]

    results = {}

    for op_name, op_func in operations.items():
        print(f"\nTesting: {op_name}")
        print(
            "Size".ljust(10)
            + "CPU (ms)".ljust(12)
            + "MPS (ms)".ljust(12)
            + "Speedup".ljust(10)
            + "Winner"
        )
        print("-" * 55)

        cpu_times = []
        mps_times = []
        speedups = []

        for size in sizes:
            try:
                # Create appropriate tensor shape based on operation
                if "Matrix multiply" in op_name:
                    # For matrix multiplication, use square matrices
                    dim = int(np.sqrt(size))
                    if dim * dim != size:
                        dim = int(np.sqrt(size)) + 1
                    tensor_shape = (dim, dim)
                    actual_size = dim * dim
                else:
                    # For other operations, use 1D tensors
                    tensor_shape = (size,)
                    actual_size = size

                # Create random tensor
                tensor = torch.randn(tensor_shape, dtype=torch.float32)

                # Benchmark on CPU
                cpu_time = benchmark_function(op_func, tensor, "cpu", num_runs=50)

                # Benchmark on MPS
                mps_time = benchmark_function(op_func, tensor, "mps", num_runs=50)

                speedup = cpu_time / mps_time
                winner = "MPS" if speedup > 1.0 else "CPU"

                cpu_times.append(cpu_time * 1000)  # Convert to ms
                mps_times.append(mps_time * 1000)  # Convert to ms
                speedups.append(speedup)

                print(
                    f"{actual_size:<10} {cpu_time*1000:<11.3f} {mps_time*1000:<11.3f} {speedup:<9.2f} {winner}"
                )

            except Exception as e:
                print(f"{size:<10} Error: {str(e)}")
                cpu_times.append(np.nan)
                mps_times.append(np.nan)
                speedups.append(np.nan)

        results[op_name] = {
            "sizes": sizes,
            "cpu_times": cpu_times,
            "mps_times": mps_times,
            "speedups": speedups,
        }

    return results


def plot_results(results):
    """Plot the benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CPU vs MPS Performance on M3 Pro", fontsize=16)

    # Plot 1: Speedup vs tensor size for different operations
    ax1 = axes[0, 0]
    for op_name, data in results.items():
        if not all(np.isnan(data["speedups"])):
            ax1.semilogx(
                data["sizes"], data["speedups"], marker="o", label=op_name, alpha=0.7
            )
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax1.set_xlabel("Tensor Size (elements)")
    ax1.set_ylabel("Speedup (CPU time / MPS time)")
    ax1.set_title("Speedup vs Tensor Size")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute times for matrix multiplication
    ax2 = axes[0, 1]
    if "Matrix multiply (square)" in results:
        data = results["Matrix multiply (square)"]
        ax2.loglog(data["sizes"], data["cpu_times"], "o-", label="CPU", alpha=0.7)
        ax2.loglog(data["sizes"], data["mps_times"], "s-", label="MPS", alpha=0.7)
        ax2.set_xlabel("Tensor Size (elements)")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Matrix Multiplication: Absolute Times")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Absolute times for element-wise operations
    ax3 = axes[1, 0]
    if "Element-wise multiply" in results:
        data = results["Element-wise multiply"]
        ax3.loglog(data["sizes"], data["cpu_times"], "o-", label="CPU", alpha=0.7)
        ax3.loglog(data["sizes"], data["mps_times"], "s-", label="MPS", alpha=0.7)
        ax3.set_xlabel("Tensor Size (elements)")
        ax3.set_ylabel("Time (ms)")
        ax3.set_title("Element-wise Multiply: Absolute Times")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Summary of crossover points
    ax4 = axes[1, 1]
    crossover_points = {}
    for op_name, data in results.items():
        speedups = np.array(data["speedups"])
        sizes = np.array(data["sizes"])

        # Find first point where MPS is consistently faster (speedup > 1.0)
        faster_indices = np.where(speedups > 1.0)[0]
        if len(faster_indices) > 0:
            crossover_points[op_name] = sizes[faster_indices[0]]

    if crossover_points:
        operations = list(crossover_points.keys())
        crossovers = list(crossover_points.values())

        y_pos = np.arange(len(operations))
        bars = ax4.barh(y_pos, crossovers, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([op.replace(" ", "\n") for op in operations])
        ax4.set_xlabel("Crossover Point (tensor elements)")
        ax4.set_title("MPS Becomes Faster At...")
        ax4.set_xscale("log")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                ha="left",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def find_crossover_summary(results):
    """Print a summary of crossover points."""
    print("\n" + "=" * 60)
    print("CROSSOVER POINT SUMMARY")
    print("=" * 60)

    for op_name, data in results.items():
        speedups = np.array(data["speedups"])
        sizes = np.array(data["sizes"])

        # Find first point where MPS is faster
        faster_indices = np.where(speedups > 1.0)[0]

        if len(faster_indices) > 0:
            crossover = sizes[faster_indices[0]]
            max_speedup = np.nanmax(speedups)
            print(f"{op_name}:")
            print(f"  - MPS faster starting at: {crossover:,} elements")
            print(f"  - Maximum speedup achieved: {max_speedup:.2f}x")
        else:
            print(f"{op_name}:")
            print(f"  - CPU remained faster at all tested sizes")
        print()


if __name__ == "__main__":
    print("Benchmarking PyTorch operations: CPU vs MPS on M3 Pro")
    print("This may take a few minutes...")

    results = test_operations()

    if results:
        find_crossover_summary(results)

        try:
            plot_results(results)
        except ImportError:
            print(
                "Matplotlib not available for plotting. Install with: pip install matplotlib"
            )
        except Exception as e:
            print(f"Plotting failed: {e}")
            print("Results printed above.")
