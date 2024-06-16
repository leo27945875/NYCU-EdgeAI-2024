import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def Main():
    MODEL_WEIGHT_PATH = "./model/311581017_model.pt"
    MODEL_PTE_PATH    = "./model/xnn_mobilenet.pte"
    
    # Load model:
    model = mobilenet_v2()
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 10)
    )
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    # XNN partition:
    args = (torch.randn(1, 3, 224, 224),)
    edgeProgram = to_edge(export(model, args))
    edgeProgram = edgeProgram.to_backend(XnnpackPartitioner)

    # Executorch:
    executorchProgram = edgeProgram.to_executorch()
    with open(MODEL_PTE_PATH, 'wb') as f:
        f.write(executorchProgram.buffer)


if __name__ == "__main__":

    Main()