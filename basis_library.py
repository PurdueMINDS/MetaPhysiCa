import numpy as np
import torch
from itertools import combinations_with_replacement


class BasisLibrary:
    """
    Library for basis functions (possibly with learnable parameters)
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.basis = []

    @property
    def n_input_features(self):
        return self.input_dim

    @property
    def n_output_features(self):
        return len(self.basis)

    def add_poly_library(self, degree):
        f_config_list = []
        for d in range(degree+1):
            f_config_list.extend(list(combinations_with_replacement(range(self.input_dim), d)))
        self.basis.extend([self._return_polyf(f_config) for f_config in f_config_list])

    def add_fourier_library(self, params=None):
        for idx in range(self.input_dim):
            self.basis.append(self._return_fourierf(idx, params))

    def transform(self, x):
        transformed_x = torch.stack([f(x) for f, _ in self.basis], dim=-1)
        return transformed_x

    def get_basis_names(self):
        return [name_f() for _, name_f in self.basis]
    
    def _return_polyf(self, f_config):
        bincount = np.bincount(f_config)

        def name_f():
            name = " ".join([f"x{idx}^{p}" if p > 1 else f"x{idx}" for idx, p in enumerate(bincount) if p > 0])
            if name == "":
                name = "1"
            return name
        
        def f(x):
            result = torch.ones_like(x.select(-1, 0))
            for idx, p in enumerate(bincount):
                result *= x.select(-1, idx) ** p
            return result

        return f, name_f

    def _return_fourierf(self, idx, params):
        if params is None or params.shape[0]==0:
            def name_f():
                return f"sin(x{idx})" 
            def f(x):
                return torch.sin(x.select(-1, idx))
        else:
            def name_f():
                return f"sin({params[idx, 0].item()} x{idx} + {params[idx, 1].item()})" 
            def f(x):
                return torch.sin(params[idx, 0] * x.select(-1, idx) + params[idx, 1])
        return f, name_f
    

if __name__ == '__main__':

    input_dim = 3
    basis_params = torch.tensor([1., 0.]).repeat((input_dim, 1))
    basis_params.requires_grad_(True)
    lib = BasisLibrary(input_dim=input_dim)
    lib.add_poly_library(degree=3)
    lib.add_fourier_library(basis_params)

    # Single input
    x = torch.tensor([0.6, 0.1, 0.1])
    out = lib.transform(x)
    print(out)
    out.sum().backward()
    print(basis_params.grad)


    # Batch of inputs
    x = torch.tensor([[0.6, 0.1, 0.1], [0.1, 0.1, 0.5]])
    out = lib.transform(x)
    print(out)

    print(lib.get_basis_names())
    basis_params.data[0] = torch.tensor([-1., -1.])
    print(lib.get_basis_names())

