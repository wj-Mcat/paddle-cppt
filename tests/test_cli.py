from __future__ import annotations
from cppt.cli import Structure


def test_is_same_layer():
    # first_name, second_name = 'aa.0.bb', 'aa.0.cc'
    # assert Structure.is_same_layer(first_name, second_name)

    first_name, second_name = 'aa.0.bb', 'bb.0.cc'
    assert not Structure.is_same_layer(first_name, second_name)

    first_name, second_name = 'aa.0.bb', 'aa.0.1.cc'
    assert not Structure.is_same_layer(first_name, second_name)
