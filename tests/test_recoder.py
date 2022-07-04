"""unit test for recorder"""
import torch
import paddle
from cppt.record import TensorRecorder, TensorComparer


def test_tensor_recorder(tmp_path):
    """test tensor recorder"""
    torch_recorder = TensorRecorder(run_id=1, name='torch', record_file=tmp_path / 'torch.json')
    paddle_reorder = TensorRecorder(run_id=1, name='paddle', record_file=tmp_path / 'paddle.json')

    torch_recorder.add('1', torch.randn(3, 4))
    paddle_reorder.add('2', paddle.to_tensor(
        [1, 2, 3, 4, 543, 24, 23423, 423, 4234]))

    comparer = TensorComparer(torch_recorder, paddle_reorder)
    comparer.compare()