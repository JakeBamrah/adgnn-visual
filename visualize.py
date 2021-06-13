from torch import tensor
from torch.utils.tensorboard import SummaryWriter


def record_amgnn_module_metrics(amgnn, writer):
    """Expects AMGNN module and tensorboard summary writer"""

    for id, l in enumerate(amgnn.gnn_obj.modules()):
        class_name = l.__class__.__name__
        if class_name == 'Wcompute':
            bias, weight = l.conv2d_last.bias, l.conv2d_last.weight
            writer.add_histogram(f'{class_name}-{id} bias', bias)
            writer.add_histogram(f'{class_name}-{id} weight', weight)

        if class_name == 'Gconv':
            bias, weight = l.fc.bias, l.fc.weight
            writer.add_histogram(f'{class_name}-{id} bias', bias)
            writer.add_histogram(f'{class_name}-{id} weight', weight)

    return writer


def build_feature_embedding_graph(data, writer, tag="AD samples"):
    """
        Build tensorboard projection of features and their respective labels.
        Data format expected as: { label: [[node_features], ..., [node_features] }
    """

    labels = []
    features = []
    for k, v in data.items():
        for n in v:
            labels.append(k)
            features.append(n)

    features = tensor(features)
    writer.add_embedding(features, metadata=labels, tag=tag)
    return
