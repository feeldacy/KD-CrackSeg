import torch
import torch.nn as nn


class MFE(nn.Module):
    """Extract and fuse features at four dilation rates over multiple layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.num_layers = num_layers
        self.dilations = (1, 3, 5, 7)
        self.input_projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        branch_layers = []
        for layer_index in range(num_layers):
            branch_in_channels = out_channels if layer_index == 0 else 5 * out_channels
            branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            branch_in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                    for dilation in self.dilations
                ]
            )
            branch_layers.append(branches)

        self.layers = nn.ModuleList(branch_layers)
        self.fusion = nn.Conv2d(
            (num_layers * len(self.dilations) + 1) * out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_features = self.input_projection(x)
        previous_layer_features = None
        layer_outputs = []

        for branches in self.layers:
            if previous_layer_features is None:
                layer_input = stem_features
            else:
                layer_input = torch.cat(
                    (stem_features, previous_layer_features), dim=1
                )

            branch_outputs = [branch(layer_input) for branch in branches]
            previous_layer_features = torch.cat(branch_outputs, dim=1)
            layer_outputs.extend(branch_outputs)

        return self.fusion(torch.cat([stem_features] + layer_outputs, dim=1))
