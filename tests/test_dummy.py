def test_imports():
    import torch
    import lightning
    import deepspeed
    import torchvision

    assert torch.__version__ is not None
    assert lightning.__version__ is not None

def test_dummy():
    assert 1 + 1 == 2
