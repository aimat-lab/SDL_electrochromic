import pandas as pd
import numpy as np
from typing import List


def table_to_numpy(path, n_obj=2):
    df = pd.read_excel(path, index_col=0)
    # shift to 0
    df["spin_speed"] = df["spin_speed"] - 500
    df["spin_time"] = df["spin_time"] - 10
    df["spin_acceleration"] = df["spin_acceleration"] - 1000

    x = df.to_numpy()[:, :4]
    
    if n_obj==2:
        y = df[['defects', 'optical_density']].to_numpy()*np.array([-1, 1]) #we want to minimize the first objective, i.e. number of particles
    elif n_obj==3:
        y = df[['defects_bright', 'defects_dark', 'optical_density']].to_numpy()*np.array([-1, -1, 1]) #we want to minimize the first and second objectives, i.e. number of particles
    
    return x, y


def closest_to(x: float, li: List[float] = [2.4, 3., 3.4, 4.]):
        dist = []
        for el in li:
            dist.append(abs(x-el))
        sel = np.argmin(dist)
        return (li[sel])


def increment_id(input_string, N):
    numeric_part = int(input_string[2:])
    new_ids = []
    for i in range(N):
        numeric_part = numeric_part + 1
        new_ids.append(f"ID{numeric_part:03}")
    return new_ids
