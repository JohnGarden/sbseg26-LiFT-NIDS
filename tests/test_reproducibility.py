import random
import unittest

import pytest

from lift_nids.utils import set_global_seed


class ReproducibilityTests(unittest.TestCase):
    def test_set_global_seed_makes_random_deterministic(self) -> None:
        set_global_seed(42)
        first_random = random.random()

        set_global_seed(42)
        second_random = random.random()

        self.assertEqual(first_random, second_random)


torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _restore_cudnn_flags():
    """Restore the global cuDNN backend flags after each test mutates them."""
    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.deterministic = prev_det
    torch.backends.cudnn.benchmark = prev_bench


def test_deterministic_false_enables_cudnn_autotuner():
    """deterministic=False must select the speed (benchmark) cuDNN mode."""
    set_global_seed(42, deterministic=False)
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False


def test_deterministic_true_is_the_default_and_forces_reproducible_mode():
    """The default (deterministic=True) must disable the autotuner, even after
    a prior call left benchmark enabled."""
    set_global_seed(1, deterministic=False)  # leave benchmark on
    set_global_seed(1)                        # default -> must flip it back
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


if __name__ == "__main__":
    unittest.main()
