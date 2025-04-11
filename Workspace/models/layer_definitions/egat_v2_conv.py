import typing
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)


class EGATv2Conv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels, bias=bias)

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if residual:
            self.res = Linear(
                in_channels if isinstance(in_channels, int) else in_channels[1],
                out_channels * heads if concat else out_channels,
                bias=False,
            )
        else:
            self.register_parameter("res", None)

        if bias:
            self.bias = Parameter(
                torch.empty(out_channels * heads if concat else out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        x = x.float()  # Ensure input is in float format
        H, C = self.heads, self.out_channels

        x_l, x_r = (
            (self.lin_l(x), self.lin_r(x)) if isinstance(x, Tensor) else (x[0], x[1])
        )
        x_l = x_l.view(-1, H, C)
        x_r = x_r.view(-1, H, C)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def edge_update(
        self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor, index: Tensor
    ) -> Tensor:
        alpha = x_i + x_j
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha += edge_attr
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
