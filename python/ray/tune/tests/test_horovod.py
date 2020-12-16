import pytest

import ray
from ray import tune

pytest.importorskip("horovod")

try:
    from ray.tune.integration.horovod import (DistributedTrainableCreator,
                                              _train_simple)
except ImportError:
    pass  # This shouldn't be reached - the test should be skipped.


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_connect_cluster():
    try:
        address_info = ray.init(address="auto")
    except Exception as e:
        pytest.skip(str(e))
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.mark.skip("TODO(Edi): Fix ImportError: horovod/torch/mpi_lib_v2.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit6tracer9addInputsEPNS0_4NodeEPKcRKN3c1013TensorOptionsE")
def test_single_step(ray_start_2_cpus):
    trainable_cls = DistributedTrainableCreator(
        _train_simple, num_hosts=1, num_slots=2)
    trainer = trainable_cls()
    trainer.train()
    trainer.stop()


@pytest.mark.skip("TODO(Edi): Fix ImportError: horovod/torch/mpi_lib_v2.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit6tracer9addInputsEPNS0_4NodeEPKcRKN3c1013TensorOptionsE")
def test_step_after_completion(ray_start_2_cpus):
    trainable_cls = DistributedTrainableCreator(
        _train_simple, num_hosts=1, num_slots=2)
    trainer = trainable_cls(config={"epochs": 1})
    with pytest.raises(RuntimeError):
        for i in range(10):
            trainer.train()


def test_validation(ray_start_2_cpus):
    def bad_func(a, b, c):
        return 1

    t_cls = DistributedTrainableCreator(bad_func, num_slots=2)
    with pytest.raises(ValueError):
        t_cls()


@pytest.mark.skip("TODO(Edi): Fix ImportError: horovod/torch/mpi_lib_v2.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit6tracer9addInputsEPNS0_4NodeEPKcRKN3c1013TensorOptionsE")
def test_set_global(ray_start_2_cpus):
    trainable_cls = DistributedTrainableCreator(_train_simple, num_slots=2)
    trainable = trainable_cls()
    result = trainable.train()
    trainable.stop()
    assert result["rank"] == 0


@pytest.mark.skip("TODO(Edi): Fix ImportError: horovod/torch/mpi_lib_v2.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit6tracer9addInputsEPNS0_4NodeEPKcRKN3c1013TensorOptionsE")
def test_simple_tune(ray_start_4_cpus):
    trainable_cls = DistributedTrainableCreator(_train_simple, num_slots=2)
    analysis = tune.run(
        trainable_cls, num_samples=2, stop={"training_iteration": 2})
    assert analysis.trials[0].last_result["training_iteration"] == 2


@pytest.mark.parametrize("use_gpu", [True, False])
def test_resource_tune(ray_connect_cluster, use_gpu):
    if use_gpu and ray.cluster_resources().get("GPU", 0) == 0:
        pytest.skip("No GPU available.")
    trainable_cls = DistributedTrainableCreator(
        _train_simple, num_slots=2, use_gpu=use_gpu)
    analysis = tune.run(
        trainable_cls, num_samples=2, stop={"training_iteration": 2})
    assert analysis.trials[0].last_result["training_iteration"] == 2


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
