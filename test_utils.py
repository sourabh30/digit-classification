from utilities import create_hparam_combo

def test_create_hparam_combo():
    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]
    param_combinations = create_hparam_combo(gamma_range, C_range)
    assert len(param_combinations) == len(gamma_range) * len(C_range)


