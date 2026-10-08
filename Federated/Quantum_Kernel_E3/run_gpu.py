import os

# Make PyTorch CUDA allocation more resistant to fragmentation
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True"
)

from flwr.simulation import run_simulation

from pytorchexample.server_app import app as server_app
from pytorchexample.client_app import app as client_app


if __name__ == "__main__":

    print("=" * 80)
    print("NEUROVERSE — QUANTUM-KERNEL E3")
    print("GPU-CONTROLLED FLOWER SIMULATION")
    print("=" * 80)

    print("Clients:              3")
    print("Clients:              C1, C2, C3")
    print("GPU resources:        1")
    print("GPU per client:       1")
    print("Client concurrency:   1 GPU client at a time")
    print("Final GlobalTest:     GPU")
    print("=" * 80)

    backend_config = {
        "init_args": {
            "num_gpus": 1,
        },
        "client_resources": {
            "num_cpus": 2,
            "num_gpus": 1.0,
        },
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=False,
    )
