import torch


def assert_model_consistency(model):
    for i, layer in enumerate(model.layers):
        in_expected = model.layer_dims[i]
        out_expected = model.layer_dims[i + 1]
        assert layer.in_features == in_expected, f"Layer {i} has incorrect in_features"
        assert (
            layer.out_features == out_expected
        ), f"Layer {i} has incorrect out_features"


def assert_output_preserved(input_tensor, operation, kwargs=None):
    kwargs = kwargs or {}
    model = kwargs.get("model")
    assert model is not None, "Model must be provided in kwargs"

    with torch.no_grad():
        before = model(input_tensor).clone()

    operation(**kwargs)

    with torch.no_grad():
        after = model(input_tensor).clone()

    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
