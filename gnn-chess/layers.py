import torch


def _scatter_sum(values, group_index, n_groups):
    return values.new_zeros((n_groups,) + values.shape[1:]).index_add_(0, group_index, values)


def _scatter_mean(values, group_index, n_groups):
    counts = _scatter_sum(torch.ones_like(values[:, :1]), group_index, n_groups).clamp_(min=1)
    return _scatter_sum(values, group_index, n_groups) / counts


def _scatter_max(values, group_index, n_groups):
    shape = (n_groups,) + values.shape[1:]
    index = group_index.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    return torch.full(shape, float('-inf'), device=values.device, dtype=values.dtype).scatter_reduce_(
        0, index, values, 'amax', include_self=True)


AGGREGATORS = {'sum': _scatter_sum, 'mean': _scatter_mean, 'max': _scatter_max}


def scatter_and_aggregate(values: torch.Tensor, group_index: torch.Tensor, n_groups: int,
                          agg_method: str = 'sum') -> torch.Tensor:
    return AGGREGATORS[agg_method](values, group_index, n_groups)


def scatter_and_softmax(scores: torch.Tensor, group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    shifted = torch.exp(scores - _scatter_max(scores, group_index, n_groups).index_select(0, group_index))
    return shifted / _scatter_sum(shifted, group_index, n_groups).index_select(0, group_index)


def concat_edges_ends_only(node_feats, receivers, senders):
    return torch.cat([node_feats.index_select(0, receivers), node_feats.index_select(0, senders)], -1)


def concat_edge_with_ends(node_feats, edge_feats, receivers, senders):
    return torch.cat([edge_feats, node_feats.index_select(0, receivers), node_feats.index_select(0, senders)], -1)


def concat_globals_with_anything(global_feats, items, position_of_item):
    return torch.cat([global_feats.index_select(0, position_of_item), items], -1)


def global_average_pooling(node_feats: torch.Tensor, n_positions: int) -> torch.Tensor:
    return node_feats.view(n_positions, 64, node_feats.shape[-1]).mean(1)
