"""
Exact Gaussian Process Regression using GPytorch
Adapted from https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html
"""
import math
import torch
import gpytorch
from tqdm import tqdm

def toy_func(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x * (2 * math.pi)) + torch.randn_like(x) * math.sqrt(0.04)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training data is 500 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 500, device="cpu")
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = toy_func(train_x)

    # Test points are regularly spaced along [0,1]
    test_x = torch.linspace(0, 1, 51, device="cpu")
    test_y = toy_func(test_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()


    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # Move the results to the CPU
    mean = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()

    # Move the training data to the CPU
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    test_y = test_y.cpu()
    rmse = (mean - test_y).square().mean().sqrt().item()
    print(f"RMSE: {rmse:.3f}")


