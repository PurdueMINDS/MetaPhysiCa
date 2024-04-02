"""
Datasets.

Usage:
  metaphysica.py --data=data [--datatype=datatype] [--polynomial_power=p] [--lr=lr] [--lambda_phi=r1] [--lambda_vrex=r2] 

Options:
  --data=data                       Dataset name
  --datatype=datatype               Dataset type (2: OOD X0; 3: OOD X0 and W*) [default: 2]
  --polynomial_power=p              Max power of polynomial basis [default: 3]
  --lr=lr                           Learning rate [default: 1e-2]
  --lambda_phi=r1                   L1 regularization strength [default: 1e-2]
  --lambda_vrex=r2                  V-REx penalty [default: 0]
"""

import numpy as np
import torch
import torch.nn as nn
import pysindy as ps
from basis_library import BasisLibrary
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torchdiffeq import odeint as nn_odeint
from datasets import *
import project_utils
from docopt import docopt

device = "cpu"

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(torch.sigmoid(input))
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * output * (1 - output)


class MetaPhysiCa(nn.Module):

    def __init__(self, **model_params):
        super().__init__()

        self.state_dim = model_params["state_dim"]
        self.is_round = model_params.get("is_round", True)
        self.polynomial_power = model_params.get("polynomial_power", 3) 

        polynomial_library = ps.PolynomialLibrary(degree=self.polynomial_power)
        fourier_library = ps.FourierLibrary(n_frequencies=1)
        self.feature_library = ps.ConcatLibrary([polynomial_library, fourier_library])

        # Change to custom basis library if basis functions are parameterized.

        # is_basis_params = model_params.get("is_basis_params", False)
        # self.feature_library = BasisLibrary(input_dim=self.state_dim)
        # self.feature_library.add_poly_library(degree=self.polynomial_power)
        # if is_basis_params:
        #     self.basis_params = nn.Parameter(torch.tensor([1., 0.]).repeat((self.state_dim, 1)))
        # else:
        #     self.basis_params = nn.Parameter(None)
        # self.feature_library.add_fourier_library(params=self.basis_params)

        # Register parameter later (after knowing the shape from data)
        self.xi = None     # Selection parameters
        self.W = None      # Coefficients for the terms selected
        self.allW = None   # All coefficients if joint fit_type
   

    def forward_single(self, transformed_y, W, xi):
        # Forward a single curve 
        if self.is_round:
            return transformed_y @ (W * STEFunction.apply(xi))
        else:
            return transformed_y @ (W * torch.sigmoid(xi))

    def predict(self, y0, t, W):

        def forward_ode_func(t, y):
            transformed_y = torch.tensor(self.feature_library.transform(y.reshape(1, -1)), dtype=torch.float32)
            return self.forward_single(transformed_y, W, self.xi.detach()).reshape(-1)

        try:
            pred_y = nn_odeint(forward_ode_func, y0, t)
        except Exception as e:
            print(f"Exception {e}")
            pred_y = nn_odeint(forward_ode_func, y0, t, method='rk4')

        return pred_y

    def compute_task_loss(self, transformed_y, dy, W):
        # Keep xi constant
        pred_dy = self.forward_single(transformed_y, W, self.xi.detach())
        loss = ((dy - pred_dy) ** 2).mean()
        return loss

    def compute_total_loss(self, transformed_y, dy, W):
        pred_dy = self.forward_single(transformed_y, W, self.xi)
        loss = ((dy - pred_dy) ** 2).mean()
        return loss

    def fit(self, train_data, **fit_params):
        y = train_data.y
        transformed_y = self.feature_library.fit_transform(y.reshape(-1, y.shape[-1]))
        transformed_y = torch.tensor(transformed_y.reshape(y.shape[0], y.shape[1], -1), dtype=torch.float32, device=device)
        diff_method = fit_params.get("diff_method", "smooth")
        dy = train_data.dy[diff_method].to(device)

        # Fit parameters
        n_inner_epochs = fit_params.get("n_inner_epochs", 1000)
        optimizer_type = fit_params.get("optimizer_type", "adam")
        xi_init_type = fit_params.get("xi_init_type", "ones")   # ones, rand, lr
        lr = fit_params.get("lr", 1e-2)

        lambda_vrex = fit_params.get("lambda_vrex", 0)
        lambda_phi = fit_params.get("lambda_phi", 1e-2)

        filename = fit_params["filename"]
        debug = fit_params.get("debug", False)
        n_batches = fit_params.get("n_batches", 1)
        batch_size = int(np.ceil(y.shape[0] / n_batches))

        if xi_init_type == "ones":
            self.xi = nn.Parameter(torch.ones(self.feature_library.n_output_features_, self.feature_library.n_features_in_, device=device))
        elif xi_init_type == "rand":
            self.xi = nn.Parameter(torch.rand(self.feature_library.n_output_features_, self.feature_library.n_features_in_, device=device)) - 0.5
        else:
            raise NotImplementedError
        self.allW = nn.Parameter(torch.rand(y.shape[0], self.feature_library.n_output_features_, self.feature_library.n_features_in_, device=device))

        # self.xi = nn.Parameter(torch.ones(self.feature_library.n_output_features, self.feature_library.n_input_features, device=device))
        # self.allW = nn.Parameter(torch.rand(y.shape[0], self.feature_library.n_output_features, self.feature_library.n_input_features, device=device))

        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        lambda_vrex_scheduled = project_utils.LinearStepScheduler(0., lambda_vrex, n_inner_epochs, n_inner_epochs//100)

        best_loss = np.inf
        best_save_loss = np.inf
        best_xi = None
        for epoch in tqdm(range(n_inner_epochs), leave=False):
            for batch_idx in range(0, y.shape[0], batch_size):
                task_losses = []
                task_reg = 0.
                for i in range(batch_idx, batch_idx + batch_size):
                    loss = self.compute_total_loss(transformed_y[i], dy[i], self.allW[i])
                    task_losses.append(loss)
                
                task_losses = torch.stack(task_losses)
                loss = task_losses.mean()

                # V-REx penalty
                task_reg = task_losses.var()
                # task_reg = task_losses.std()

                # L1
                l1_reg_xi = torch.norm(STEFunction.apply(self.xi), 1)
                total_loss = loss + lambda_phi * l1_reg_xi + lambda_vrex_scheduled.get() * task_reg

                saved = ""
                save_loss = (loss + lambda_phi * l1_reg_xi + lambda_vrex * task_reg).item()
                if save_loss < best_save_loss:
                    best_save_loss = save_loss
                    best_loss = loss.item()
                    best_xi = self.xi.data.clone()
                    best_allW = self.allW.data.clone()
                    saved = "(saved)"

                if debug:
                    print("="*40)
                    self.print()
                    print(loss, task_reg, l1_reg_xi, lambda_vrex_scheduled.get(), saved)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            lambda_vrex_scheduled.step()

        print(f"Fit complete with final loss: {best_loss}, {best_save_loss}")
        self.xi.data = best_xi.data
        self.allW.data = best_allW.data
        print(self.xi)
        self.print()
        torch.save(self, filename + ".pkl")

        return {"loss": best_loss, "sparsity": (self.xi > 0).sum().item()}


    def test(self, test_data, test_filename=None, **test_params):
        test_t = test_data.t
        test_y = test_data.y
        input_length = test_data.input_length

        test_transformed_y = self.feature_library.fit_transform(test_y.reshape(-1, test_y.shape[-1]))
        test_transformed_y = torch.tensor(test_transformed_y.reshape(test_y.shape[0], test_y.shape[1], -1), dtype=torch.float32, device=device)
        test_transformed_y = test_transformed_y[:, :input_length]
        diff_method = test_params.get("diff_method", "smooth")
        test_dy = test_data.dy[diff_method].to(device)[:, :input_length]

        # Test adapt params
        n_inner_epochs = test_params.get("n_test_inner_epochs", 1000)
        test_lr = test_params.get("test_lr", 1e-3)
        debug = test_params.get("debug", False)
        plot = test_params.get("plot", None)

        # Test-time adapt
        test_pred_y = []
        
        meanW = self.allW.detach().mean(dim=0)
        for i in tqdm(range(test_y.shape[0]), leave=False):
            # Separate task-specific parameters for test tasks
            # self.W = nn.Parameter(torch.rand(model.feature_library.n_output_features_, model.feature_library.n_input_features_, device=device))
            self.W = nn.Parameter(meanW.clone())

            optimizer = torch.optim.Adam([self.W], lr=test_lr)

            all_losses = []
            best_loss = np.inf
            best_W = None
            for epoch in range(n_inner_epochs):
                optimizer.zero_grad()
                loss = self.compute_task_loss(test_transformed_y[i], test_dy[i], self.W)        # Compute loss keeping xi fixed
                all_losses.append(loss.item())
                total_loss = loss

                if total_loss < best_loss:
                    best_loss = total_loss.item()
                    best_W = self.W.data.clone()

                total_loss.backward()
                optimizer.step()

            all_losses = np.array(all_losses)

            try:
                test_pred_y.append(self.predict(test_y[i, 0], test_t, W=best_W))
            except Exception as e:
                print(e)
                test_pred_y.append(torch.tensor(np.nan * np.ones_like(test_y[i])))

            if debug:
                print("=" * 80)
                print(best_W * STEFunction.apply(self.xi))
                plt.plot(range(n_inner_epochs//10, n_inner_epochs), all_losses[n_inner_epochs//10:])
                plt.show()
                plt.plot(test_t, test_y[i, :, 0].detach().cpu().numpy())
                plt.plot(test_t, test_pred_y[i][:, 0].detach().cpu().numpy())
                plt.axvline(test_t[input_length], color='black', linestyle='--')
                plt.show()

        test_pred_y = torch.stack(test_pred_y)
        
        if plot:
            dim = test_y.shape[-1]
            ylabels = test_data.state_names
            labels = ["True", "Predicted"]
            markers = ["o", "^"]
            n_plots = 5

            # Plot random curves
            indices = np.random.permutation(test_y.shape[0])[:n_plots]

            for j, idx in enumerate(indices):
                y_list = [test_y[idx], test_pred_y[idx]]

                fig, ax = plt.subplots(1, dim, figsize=(12//2 * dim, 9//2))
                for d in range(dim):
                    for i, X in enumerate(y_list):
                        ax[d].plot(test_t, X[:, d], label=labels[i], marker=markers[i]) 
                    ax[d].axvline(
                        test_t[input_length],
                        color='black',
                        linestyle='--',
                    )
                    ax[d].set_xlabel("t", fontsize=18)
                    ax[d].set_ylabel(ylabels[d], fontsize=18)
                    ax[d].legend()

                if plot == "save" or plot == "both":
                    plt.savefig(f"{test_filename}_{j}.pdf", dpi=300)
                if plot == "show" or plot == "both":
                    plt.show()
                plt.close()

        metrics = project_utils.get_all_metrics(test_pred_y, test_y, input_length)
        metrics["sparsity"] = (self.xi > 0).sum().item()
        return metrics

    def print(self):
        if self.xi is None:
            print("No coefficients")
            return
        else:
            coef = torch.round(torch.sigmoid(self.xi)).detach().cpu().numpy()
            input_features = self.feature_library.get_feature_names()
            for j in range(coef.shape[1]):
                rhs = [f"{input_features[i]}" for i, c in enumerate(coef[:, j]) if np.abs(c) > 0.]
                print(f"x{j}' = " + " + ".join(rhs))


def run(train_data, test_data, _config, _run):
    """
    Run model on given train, test data and compute metrics
    """
    # Use 20% of the training data as validation (for hyperparameter tuning)
    trainIdx, validIdx = project_utils.getRandomSplit(len(train_data), [80, 20])

    valid_data = SubsetStar(train_data, validIdx)
    train_data = SubsetStar(train_data, trainIdx)

    train_results = None
    if "fit" not in _config or _config["fit"]:
        model = MetaPhysiCa(**_config).to(device)
        train_results = model.fit(train_data, **_config)

    model = torch.load(_config["filename"] + ".pkl").to(device)
    print(model.xi)
    model.print()

    print("Validation data")
    valid_results = model.test(valid_data, **{**_config, "plot": False})

    print("Test data: In-distribution and Out-of-distribution")
    test_results = {}
    for key, test_data_ in test_data.items():
        test_results[key] = model.test(
            test_data_, test_filename=_config["filename"] + f"_test={key}", **_config
        )

    return model, train_results, valid_results, test_results


if __name__ == '__main__':
    args = docopt(__doc__)
    data = args["--data"]
    datatype = int(args["--datatype"])

    if data == "damped_pendulum":
        data_kfold = DampedPendulumDataset.get_standard_dataset(root='./data', datatype=datatype, n_samples=1000)
    elif data == "lotka_volterra":
        data_kfold = LotkaVolterraDataset.get_standard_dataset(root='./data', datatype=datatype, n_samples=1000)
    elif data == "sir":
        data_kfold = SIREpidemicDataset.get_standard_dataset(root='./data', datatype=datatype, n_samples=1000)
    else:
        raise NotImplementedError

    # Running on a single CV fold.
    train_data, id_test_data, ood_test_data = data_kfold[0]

    params = {
        "state_dim": train_data.state_dim,
        "is_round": True,
        "diff_method": "smooth",
        "polynomial_power": int(args["--polynomial_power"]),

        # Fit params
        "n_inner_epochs": 10000,
        "lr": float(args["--lr"]),
        "lambda_vrex": float(args["--lambda_vrex"]),
        "lambda_phi": float(args["--lambda_phi"]),
        "n_batches": 1,

        # Test params
        "n_test_inner_epochs": 10000,
        "test_lr": 1e-3,
        "plot": "show",

        "filename": f"results/metaphysica_{train_data.__class__.__name__}",
    }

    train_data.plot()
    test_data = {"id": id_test_data, "ood": ood_test_data}
    model, train_results, valid_results, test_results = run(train_data, test_data, params, None)

    print(f"Test ID NRMSE: {test_results['id']['nrmse']:.4f}")
    print(f"Test OOD NRMSE: {test_results['ood']['nrmse']:.4f}")


