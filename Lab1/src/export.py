import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from torch._export import capture_pre_autograd_graph
from torch.export import export
from executorch.exir import EdgeCompileConfig, to_edge


def Main():
    
    MODEL_WEIGHT_PATH = "./model/311581017_model.pt"
    MODEL_PTE_PATH    = "./model/311581017_model.pte"
    
    # Load model:
    model = mobilenet_v2()
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 10)
    )
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location="cpu"))
    model.eval()

    # ATen dialect:
    args = (torch.randn(1, 3, 224, 224),)
    atenDialectPreAutograd = capture_pre_autograd_graph(model, args)
    atenDialect = export(atenDialectPreAutograd, args)

    # Edge dialect:
    edgeProgram = to_edge(atenDialect)

    # Executorch:
    executorchProgram = edgeProgram.to_executorch()
    with open(MODEL_PTE_PATH, 'wb') as f:
        f.write(executorchProgram.buffer)


if __name__ == "__main__":

    Main()