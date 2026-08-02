import unittest

from lift_nids.utils import is_valid_experiment_name


class ExperimentNamingTests(unittest.TestCase):
    def test_experiment_name_convention_accepts_valid_name(self) -> None:
        self.assertTrue(is_valid_experiment_name("exp_001_ciciot_static_xgb"))

    def test_experiment_name_convention_rejects_invalid_name(self) -> None:
        self.assertFalse(is_valid_experiment_name("experiment_001_invalid"))


if __name__ == "__main__":
    unittest.main()
