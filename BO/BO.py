import numpy as np
import torch
from typing import Dict

from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from utils import table_to_numpy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model(problem, train_x, train_obj):
    train_x = normalize(train_x, problem.bounds)
    train_y = train_obj
    models = []
    for i in range(train_y.shape[-1]):
        models.append(SingleTaskGP(train_x, train_y[..., i: i + 1], outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_qnehvi_and_get_observation(problem, model, train_x, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""

    CANDIDATES = 13
    NUM_RESTARTS = 10
    RAW_SAMPLES = 512

    standard_bounds = torch.zeros(2, problem.dim, dtype=torch.double, device=DEVICE)
    standard_bounds[1] = 1

    train_x = normalize(train_x, problem.bounds)
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objectives))),
    )
    # optimize
    print(f'Optimizing acquisition function...')
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=CANDIDATES,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    # return new suggestions
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    return new_x


def run_BO(
        problem, 
        initial_data_path: str,
        results_path: str = '',
        mc_samples: int = 128,
) -> Dict:
    
    # global results
    hv = Hypervolume(ref_point=problem.ref_point)
    hvs = []

    # load training data and initialize model
    x, obj = table_to_numpy(initial_data_path, n_obj=problem.num_objectives)        
    train_x = torch.from_numpy(x)
    train_obj = torch.from_numpy(obj)
    print(f'Initialize model with {x.shape[0]} data points')
    mll, model = initialize_model(problem, train_x, train_obj)

    # compute pareto front
    if train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(train_obj)
        pareto_y = train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
    else:
        volume = 0.0
    hvs.append(volume)
    print(f'hypervolume: {hvs}')
    
    results = {'data': torch.cat([train_x, train_obj], dim=-1).cpu().numpy(), 'pareto_mask': pareto_mask.cpu().numpy(), 'hvs': hvs}
    for k in ['data', 'pareto_mask']:
        np.save(results_path+'BO_results_' + k + '.npy', results[k])

    # run one round of BayesOpt 
    # fit the models
    print('start fitting...')
    fit_gpytorch_mll(mll)
    
    # optimize acquisition functions and return suggestions
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    new_x = optimize_qnehvi_and_get_observation(problem, model, train_x, sampler)
    return new_x


# if __name__=='__main__':
#     import pandas as pd
    
#     hyper_parameters = {
#         "ink_concentration": (2.2, 4.2),
#         "spin_speed": (0, 5599),
#         "spin_time": (0, 114),
#         "spin_acceleration": (0, 29499),
#     }
#     prob_dim = len(hyper_parameters)
#     n_samples = 10


#     bounds = torch.empty((2, prob_dim)).float()
#     for i, (low, up) in enumerate(hyper_parameters.values()):
#         bounds[0, i] = low
#         bounds[1, i] = up
        
#     samples = draw_sobol_samples(bounds=bounds, n=n_samples, q=1).squeeze(1)

#     table = dict()
#     for sample_idx, sample in enumerate(samples):
#         sample_dict = {**dict(zip(list(hyper_parameters.keys()), sample))}
#         sample_dict["ink_concentration"] = closest_to(sample_dict["ink_concentration"].item())
#         sample_dict["spin_speed"] = 500 + 100*(sample_dict["spin_speed"].item()//100)
#         sample_dict["spin_time"] = 10 + 5*(sample_dict["spin_time"].item()//5)
#         sample_dict["spin_acceleration"] = 1000 + 500*(sample_dict["spin_acceleration"].item()//500)
#         table[int(2*sample_idx)] = sample_dict
#         table[int(2*sample_idx + 1)] = sample_dict

#     df = pd.DataFrame.from_dict(table, orient='index')
    # df.to_csv('./FLAIM_samples_database.csv')
