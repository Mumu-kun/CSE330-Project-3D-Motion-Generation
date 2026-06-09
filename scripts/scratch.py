import sys
from pathlib import Path

import torch
from torchview import draw_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.models import MotionHistoryEncoder
from utils.config import Config


def main():
    config = Config()
    encoder = MotionHistoryEncoder(config.encoder_config)
    encoder.eval()

    motion = torch.randn(1, 10, 271)
    text_emb = torch.zeros(1, 512)

    graph = draw_graph(
        encoder,
        input_data=(motion, text_emb),
        # graph_dir="LR",
        expand_nested=True,  # cleaner architecture view
        depth=3,
        hide_inner_tensors=True,
        hide_module_functions=True,
        roll=True,
    )

    g = graph.visual_graph

    # Graph layout
    g.graph_attr.update(
        rankdir="TB",
        ranksep="0.8",
        nodesep="0.25",
        dpi="300",
        splines="polyline",
        fontname="Arial",
    )

    g.node_attr.update(
        shape="box",
        style="rounded",
        fontsize="10",
        fontname="Arial",
    )

    # Edge appearance
    g.edge_attr.update(
        arrowsize="0.8",
        penwidth="1.2",
    )

    g.render(
        "encoder_architecture",
        format="svg",
        cleanup=True,
    )

    g.render(
        "encoder_architecture",
        format="png",
        cleanup=True,
    )


if __name__ == "__main__":
    main()
