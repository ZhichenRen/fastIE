import torch


def get_device_of(tensor):
    """This function returns the device of the tensor refer to
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.

    Arguments:
        tensor {tensor} -- tensor

    Returns:
        int -- device
    """

    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size, device):
    """This function returns a range vector with the desired size, starting at
    0 the CUDA implementation is meant to avoid copy data from CPU to GPU refer
    to https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.

    Arguments:
        size {int} -- the size of range
        device {int} -- device

    Returns:
        torch.Tensor -- range vector
    """

    if device > -1:
        return torch.cuda.LongTensor(size,
                                     device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def flatten_and_batch_shift_indices(indices, sequence_length):
    """This function returns a vector that correctly indexes into the flattened
    target, the sequence length of the target must be provided to compute the
    appropriate offsets. refer to
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.

    Arguments:
        indices {tensor} -- index tensor
        sequence_length {int} -- sequence length

    Returns:
        tensor -- offset index tensor
    """

    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise RuntimeError(
            'All elements in indices should be in range (0, {})'.format(
                sequence_length - 1))
    offsets = get_range_vector(indices.size(0),
                               get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target, indices, flattened_indices=None):
    """This function returns selected values in the target with respect to the
    provided indices, which have size ``(batch_size, d_1, ..., d_n,
    embedding_size)`` refer to
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.

    Arguments:
        target {torch.Tensor} -- target tensor
        indices {torch.LongTensor} -- index tensor

    Keyword Arguments:
        flattened_indices {Optional[torch.LongTensor]} -- flattened index tensor (default: {None})

    Returns:
        torch.Tensor -- selected tensor
    """

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets
