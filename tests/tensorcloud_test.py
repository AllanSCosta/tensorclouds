import tensorclouds


def test_zeros():
    cloud = tensorclouds.TensorCloud.zeros("0e + 1e", (10, 10))
    assert len(cloud) == 10
