"""
Datasets.

Usage:
  datasets.py [--root=root] --data=data 

Options:
  --root=root      Root directory [default: ./data]
  --data=data      Dataset name
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint as scipy_odeint
from torch.utils.data import Dataset
from abc import ABC
import os
from hashlib import sha1
import json
from docopt import docopt
import pysindy as ps
import project_utils

def estimate_derivatives_single_(i, y, differentiation_method, t):
    print(f"{i},")
    y_i = y[i].numpy()
    return differentiation_method._differentiate(y_i, t)

class SeriesDataset(ABC, Dataset):
    """
    Abstract class for Time Series Datasets
    y, t
    """

    def __init__(self, max_for_scaling=None):
        # y shape: (n_samples, time_steps, dimension)
        # t shape: (time_steps)

        self.state_dim = None
        self.state_names = None
        self.y = None
        self.dy = None      # Estimated derivatives
        self.t = None
        self.input_length = None
        self.max_for_scaling = max_for_scaling
        self.phy_params = None      # Fitted parameters

    def plot(self, dim=0, **kwargs):
        unscaled_y = self.return_unscaled_y()
        for i in range(len(self)):
            plt.plot(
                self.t.numpy(),
                unscaled_y[i, :, dim].numpy(),
                label=f"y(t): dimension {dim}",
            )
        if self.input_length > 0:
            plt.axvline(
                x=self.t.numpy()[self.input_length - 1], linestyle="--", color="black"
            )

        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            plt.ylabel(kwargs["ylabel"])
        if "title" in kwargs:
            plt.title(kwargs["title"])
        plt.show()

    def scale(self, is_scale=False):
        if self.max_for_scaling is None:
            if is_scale:
                self.max_for_scaling = self.y.amax(dim=[0, 1]) / 10.
            else:
                self.max_for_scaling = torch.ones(self.state_dim)

        self.y = self.y / self.max_for_scaling

    def return_unscaled_y(self):
        return self.y * self.max_for_scaling


    def estimate_derivatives(self, method="smooth"):
        if self.y is None:
            return

        t = self.t.numpy()
        if method == "tvr":
            differentiation_method = project_utils.DiffTVR(t, 0.2)
        elif method == "smooth":
            differentiation_method = ps.SmoothedFiniteDifference(order=2, smoother_kws={'window_length': 5})
        else:
            differentiation_method = ps.FiniteDifference(order=2)

        # Sequential
        dy = []
        for i in range(self.y.shape[0]):
            y_i = self.y[i].numpy()
            dy.append(differentiation_method._differentiate(y_i, t))
        dy = torch.tensor(np.stack(dy))

        return dy
    
    def estimate_all_derivatives(self):
        self.dy = {}
        self.dy["smooth"] = self.estimate_derivatives(method="smooth")
        # self.dy["tvr"] = self.estimate_derivatives(method="tvr")
    
    def get_initial_value_array(self, y0, n_samples):
        initial_value_array = []
        for i in range(self.state_dim):
            if isinstance(y0[i], tuple):
                array = np.random.uniform(*y0[i], n_samples)
            else:
                array = np.tile(y0[i], n_samples)
            initial_value_array.append(array)

        initial_value_array = np.stack(initial_value_array, axis=1)
        return initial_value_array

    def get_param_arrays(self, params, n_samples):
        param_arrays = []
        for param in params:
            if isinstance(param, tuple):
                param_array = np.random.uniform(*param, n_samples)
            else:
                param_array = np.tile(param, n_samples)
            param_arrays.append(param_array)

        return param_arrays

    def save(self):
        with open(self.save_filename, "wb") as f:
            all_var = [
                self.state_names,
                self.state_dim,
                self.input_length,
                self.t,
                self.y,
                self.dy,
                self.max_for_scaling,
                self.phy_params,
            ]
            torch.save(all_var, f)

    def load(self):
        print(f"Using saved file: {self.save_filename}")
        (
            self.state_names,
            self.state_dim,
            self.input_length,
            self.t,
            self.y,
            self.dy,
            self.max_for_scaling,
            self.phy_params,
        ) = torch.load(self.save_filename)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return idx, self.y[idx]



class SubsetStar(SeriesDataset):
    """
    Subset of a dataset at specified indices.
    Extended version of what is implemented in Pytorch.

    Arguments:
        dataset (SeriesDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        super().__init__()
        self.y = dataset.y[indices]
        self.t = dataset.t
        self.state_names = dataset.state_names
        self.state_dim = dataset.state_dim
        self.input_length = dataset.input_length
        self.max_for_scaling = dataset.max_for_scaling

        self.phy_params = dataset.phy_params[indices] if dataset.phy_params is not None else None
        if dataset.dy is None:
            self.dy = None
        else:
            self.dy = {}
            for k in dataset.dy.keys():
                self.dy[k] = dataset.dy[k][indices]


class Concat(SeriesDataset):
    """
    Concat two/more Series Datasets
    """

    def __init__(self, dataset1, dataset2):
        super().__init__()
        ## Assumes there are common attributes like t, etc.
        self.y = torch.cat([dataset1.y, dataset2.y])
        self.t = dataset1.t
        self.state_names = dataset1.state_names
        self.state_dim = dataset1.state_dim
        self.input_length = dataset1.input_length
        self.max_for_scaling = torch.maximum(dataset1.max_for_scaling, dataset2.max_for_scaling)

        if dataset1.phy_params is None:
            self.phy_params = None
        else:
            self.phy_params = torch.cat([dataset1.phy_params, dataset2.phy_params])
            
        if dataset1.dy is None:
            self.dy = None
        else:
            self.dy = {}
            for k in dataset1.dy.keys():
                self.dy[k] = torch.cat([dataset1.dy[k], dataset2.dy[k]])


class DampedPendulumDataset(SeriesDataset):
    """
    Generate damped pendulum data
    $\frac{d^2\theta}{dt^2} + \omega_0^2 \sin(\theta) + \alpha \frac{d\theta}{dt} = 0$
    where $\omega_0 = 2\pi/T_0$, and $T_0$ is the time period.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(0, 1), (0, 1)],
        omega0=1.0,
        alpha=0.2,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root="./data",
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        # Create files to save the dataset (load if already present)
        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            omega0,
            alpha,
            is_scale,
            seed,
        ]
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$\theta$', r'$\omega$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        # ODE function to generate the data from
        def damped_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * dtheta)

        # Sample initial values from the given range
        y0_array = self.get_initial_value_array(y0, n_samples)

        # Sample parameter values from the given range
        param_arrays = self.get_param_arrays([omega0, alpha], n_samples)
        omega0_array = param_arrays[0]
        alpha_array = param_arrays[1]

        # Generate the data
        self.y = [None] * n_samples
        for i in range(n_samples):
            self.y[i] = scipy_odeint(
                damped_pendulum_ode_func,
                y0_array[i],
                t,
                args=(omega0_array[i], alpha_array[i]),
            )

        self.y = torch.FloatTensor(np.stack(self.y))

        # Max scale the data if required
        self.scale(is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.estimate_all_derivatives()
        self.save()     # Save the data into the file.

    @classmethod
    def get_standard_dataset(cls, root, datatype, n_samples=100, input_length_factor=3, reload=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        for seed in range(k):
            np.random.seed(seed)    # Set seed
            if datatype == 1:
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(0, np.pi / 2), (0.0, 0.0)]
                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_id = omega0_ood = 1.0
                alpha_id = alpha_ood = 0.2

            elif datatype == 2:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(0, np.pi / 2), (0.0, -1.0)]
                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_id = omega0_ood = (1.0, 2.0)
                alpha_id = alpha_ood = (0.2, 0.4)
            
            elif datatype == 3:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id =[(0, np.pi / 2), (0.0, -1.0)]
                omega0_id = (1.0, 2.0)
                alpha_id = (0.2, 0.4)

                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_ood = (2.0, 3.0)
                alpha_ood = (0.4, 0.6)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                omega0=omega0_id,
                alpha=alpha_id,
                seed=seed,
                root=root,
                reload=reload,
            )
            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                omega0=omega0_id,
                alpha=alpha_id,
                seed=seed,
                root=root,
                reload=reload,
            )
            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                omega0=omega0_ood,
                alpha=alpha_ood,
                seed=seed,
                root=root,
                reload=reload,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold


class LotkaVolterraDataset(SeriesDataset):
    """
    Generate LV data
    Prey:
          dx/dt = \alpha * x  - \beta * x * y
    Predator
          dy/dt = \delta * x * y - \gamma * y
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(1000, 2000), (10, 20)],
        alpha=0.1 * 12,
        beta=0.005 * 12,
        gamma=0.04 * 12,
        delta=0.00004 * 12,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=".",
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            alpha,
            beta,
            gamma,
            delta,
            is_scale,
            seed,
        ]
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$x$', r'$y$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        def lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        y0_array = self.get_initial_value_array(y0, n_samples)

        param_arrays = self.get_param_arrays([alpha, beta, gamma, delta], n_samples)
        alpha_array = param_arrays[0]
        beta_array = param_arrays[1]
        gamma_array = param_arrays[2]
        delta_array = param_arrays[3]

        self.y = [None] * n_samples
        for i in range(n_samples):
            args = (
                alpha_array[i],
                beta_array[i],
                gamma_array[i],
                delta_array[i],
            )
            self.y[i] = scipy_odeint(lotka_volterra_ode_func, y0_array[i], t, args=args)

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.estimate_all_derivatives()
        self.save()
    
    @classmethod
    def get_standard_dataset(cls, root, datatype, n_samples=100, input_length_factor=3, reload=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)    # Set seed
            if datatype == 1:
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = 0.1 * 12
                beta_id = beta_ood = 0.005 * 12
                gamma_id = gamma_ood = 0.04 * 12
                delta_id = delta_ood = 0.00004 * 12

            elif datatype == 2:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)
            
            elif datatype == 3:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]

                alpha_id = (0.1 * 12, 0.2 * 12)
                beta_id = (0.005 * 12, 0.01 * 12)
                gamma_id = (0.04 * 12, 0.08 * 12)
                delta_id = (0.00004 * 12, 0.00008 * 12)
                alpha_ood = (0.2 * 12, 0.3 * 12)
                beta_ood = (0.01 * 12, 0.015 * 12)
                gamma_ood = (0.08 * 12, 0.12 * 12)
                delta_ood = (0.00008 * 12, 0.00012 * 12)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                alpha=alpha_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                delta=delta_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold


class SIREpidemicDataset(SeriesDataset):
    """
    Generate SIR epidemic data
        ds/dt = -\beta is/(s + i + r)
        di/dt = \beta is/(s + i + r) - \gamma i
        dr/dt = \gamma i
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(90, 100), (0, 5), (0, 0)],
        beta=4,
        gamma=0.4,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=".",
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            beta,
            gamma,
            is_scale,
            seed,
        ]
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 3
        self.state_names = [r'$S$', r'$I$', r'$R$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        def sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r) - gamma * i, 
                gamma * i
            )

        y0_array = self.get_initial_value_array(y0, n_samples)

        param_arrays = self.get_param_arrays([beta, gamma], n_samples)
        beta_array = param_arrays[0]
        gamma_array = param_arrays[1]

        self.y = [None] * n_samples
        for i in range(n_samples):
            args = (
                beta_array[i],
                gamma_array[i],
            )
            self.y[i] = scipy_odeint(sir_ode_func, y0_array[i], t, args=args)

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.estimate_all_derivatives()
        self.save()

    @classmethod
    def get_standard_dataset(cls, root, datatype, n_samples=100, reload=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // 5)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)    # Set seed
            if datatype == 1:
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]
                beta_id = beta_ood = 4
                gamma_id = gamma_ood = 0.4

            elif datatype == 2:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]
                beta_id = beta_ood = (4, 8)
                gamma_id = gamma_ood = (0.4, 0.8)
            
            elif datatype == 3:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]

                beta_id = (4, 8)
                gamma_id = (0.4, 0.8)
                beta_ood = (8, 12)
                gamma_ood = (0.8, 1.2)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                beta=beta_id,
                gamma=gamma_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                beta=beta_id,
                gamma=gamma_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold


def generate_all_datasets(cls, root, reload=False):
    for datatype in range(1, 4):
        print(f"Data type: {datatype}")
        data_kfold = cls.get_standard_dataset(root=root, datatype=datatype, n_samples=1000, reload=reload)
        train_data, id_test_data, ood_test_data = data_kfold[0]
        print("="*20 + "Train data" + "="*20)
        for d in range(train_data.state_dim):
            train_data.plot(dim=d, title=f"Dim: {d}")
        print("="*20 + "ID Test data" + "="*20)
        for d in range(train_data.state_dim):
            id_test_data.plot(dim=d, title=f"Dim: {d}")
        print("="*20 + "OOD Test data" + "="*20)
        for d in range(train_data.state_dim):
            ood_test_data.plot(dim=d, title=f"Dim: {d}")


if __name__ == "__main__":
    args = docopt(__doc__)
    root = args["--root"]
    data = args["--data"]

    if data == "damped_pendulum":
        generate_all_datasets(DampedPendulumDataset, root, reload=True)
        # data_kfold = DampedPendulumDataset.get_standard_dataset(root=root, datatype=1, n_samples=1000, reload=True)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # train_data.plot()
        # ood_test_data.plot()
    elif data == "lotka_volterra":
        generate_all_datasets(LotkaVolterraDataset, root, reload=True)
        # data_kfold = LotkaVolterraDataset.get_standard_dataset(root=root, datatype=3, n_samples=1000, reload=False)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # ood_test_data.plot()
    elif data == "sir":
        generate_all_datasets(SIREpidemicDataset, root, reload=True)
        # data_kfold = SIREpidemicDataset.get_standard_dataset(root=root, datatype=3, n_samples=1000, reload=False)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # ood_test_data.plot()
    else:
        raise NotImplementedError
