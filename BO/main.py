import pandas as pd
import torch
import torch.nn as nn

from utils import closest_to, increment_id
from BO import run_BO


class My_HypOpt(nn.Module):
    def __init__(self, hyppar, ref_point, n_obj=2):
        super().__init__()
        self.hyppar = [*hyppar.keys()]
        self.register_buffer('dim', torch.tensor([len(self.hyppar)]))
        self.register_buffer('num_objectives', torch.tensor([n_obj]))
        self.register_buffer('ref_point', ref_point)
        self.init_bounds(hyppar)

    def init_bounds(self, hyppar):
        bounds = torch.empty((2, self.dim)).float()
        for i, (low, up) in enumerate(hyppar.values()):
            bounds[0, i] = low
            bounds[1, i] = up
        self.register_buffer('bounds', bounds)

    def create_table(self, x):
        samples = {}
        for i in range(x.shape[0]):
            hp_values = x[i]
            sample_dict = {**dict(zip(self.hyppar, hp_values))}
            # shift back
            sample_dict["ink_concentration"] = closest_to(sample_dict["ink_concentration"].item())
            sample_dict["spin_speed"] = 500 + 100*(sample_dict["spin_speed"].item()//100)
            sample_dict["spin_time"] = 10 + 5*(sample_dict["spin_time"].item()//5)
            sample_dict["spin_acceleration"] = 1000 + 500*(sample_dict["spin_acceleration"].item()//500)
            samples[i] = sample_dict
        return samples


if __name__ == '__main__':    
    N_OBJ = 3
    PREV_BATCH = 8
    NEW_BATCH = PREV_BATCH + 1
    HYPPAR = {
        "ink_concentration": (2.2, 4.2),
        "spin_speed": (0, 5599),
        "spin_time": (0, 114),
        "spin_acceleration": (0, 29499),
    }

    if N_OBJ==2:
        REF_POINT = torch.tensor([-.10, .40]) # 2 OBJECTIVES
    if N_OBJ==3:
        REF_POINT = torch.tensor([-.9, -0.03, .40]) # 3 OBJECTIVES
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MC_SAMPLES = 256

    problem = My_HypOpt(HYPPAR, REF_POINT, N_OBJ).to(DEVICE).double()
    next_x = run_BO(
        problem,
        results_path = './',
        mc_samples = MC_SAMPLES,
        initial_data_path = f'./batch{PREV_BATCH}.xlsx', ## if from scratch then set 'initial_data_path = None'
    ) 

    suggestions = problem.create_table(next_x)
    suggestions = pd.DataFrame.from_dict(suggestions, orient='index')
    
    suggestions.sort_values(by='ink_concentration', inplace=True)
    suggestions = suggestions.reset_index(level=0, drop=True)

    suggestions_duplicate = suggestions.copy()
    suggestions = pd.concat([suggestions, suggestions_duplicate], keys=['original', 'copy']).sort_index(level=1).reset_index(level=0, drop=True)

    old_df = pd.read_excel(f'./batch{PREV_BATCH}.xlsx', index_col=0)
    new_ids = increment_id(old_df.index[-1], len(suggestions))
    suggestions.insert(0, 'sample_id', new_ids)
    suggestions.set_index('sample_id')
    suggestions.to_excel(f'./batch{NEW_BATCH}_suggestions.xlsx', index=False)


