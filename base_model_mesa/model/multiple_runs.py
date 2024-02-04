"""
This script can be used to run multiple parametrizations of the model. For the
results in the report, change the variable batch_id of the function 
multiple_runs AND in line 90 to:
    - 25_1: open exploration (section 4.3.1)
    - 25_pol: policies on adjusted uncertainty space (section 4.1.2)
    - 26_pol: second policy run (section 4.3.3)
    - final_pol: corrected code run (appendix B)
    - test: test run, relatively small sample (default)
"""

import time
import pandas as pd
import numpy as np
from model import AdaptationModel
from esdl_to_ema import samples
import matplotlib.pyplot as plt
import seaborn as sns
import dill
from pathlib import Path
def multiple_runs(in_file, batch_id = "test", save_agents = False):
    print(f"batch id is {batch_id}; starting....")

    start_init = time.time()
    sam = samples(input_file=in_file)

    end_init = time.time()
    print(f"initialisation took {np.round(end_init - start_init, 3)} seconds")
    timesteps = 50
    print(f"Running a total of {sam.num_samples} x {sam.num_runs_per_scen} = "
          f"{sam.num_samples*sam.num_runs_per_scen} samples")
    res = np.zeros(sam.num_samples * sam.num_runs_per_scen)
    outcomes = {
        "total_adapted": np.zeros_like(res),
        "utilities_adapted": np.zeros_like(res),
        "barrier_adapted": np.zeros_like(res),
        "drainage_adapted": np.zeros_like(res),
        "final_costs": np.zeros_like(res),
        "total_house_size": np.zeros_like(res),
        "start_savings": np.zeros_like(res),
        "final_savings": np.zeros_like(res)

    }

    start_time = time.time()
    index_used = {}
    for i in range(sam.num_runs_per_scen):
        for j in range(sam.num_samples):
            model = AdaptationModel(settings=sam.samples, settings_ind=j)
            for step in range(timesteps):
                model.step()

            run_ind = j+sam.num_samples*i
            agent_data = model.datacollector.get_agent_vars_dataframe()
            model_data = model.datacollector.get_model_vars_dataframe()
            Path(f"full_results_{batch_id}/run{run_ind}").mkdir(parents=True, exist_ok=True)
            if save_agents:
                with open(f"full_results_{batch_id}/run{run_ind}/agent_data", "wb") as file:
                    dill.dump(agent_data, file)

            with open(f"full_results_{batch_id}/run{run_ind}/model_data", "wb") as file:
                dill.dump(model_data, file)

            index_used[f"run{run_ind}"] = j
            outcomes["total_adapted"][run_ind] = model_data["total_adapted_households"].iloc[-1]
            outcomes["utilities_adapted"][run_ind] = model_data["total_adapted_utilities"].iloc[-1]
            outcomes["barrier_adapted"][run_ind] = model_data["total_adapted_barrier"].iloc[-1]
            outcomes["drainage_adapted"][run_ind] = model_data["total_adapted_drainage"].iloc[-1]
            outcomes["final_costs"][run_ind] = model_data["total_costs"].iloc[-1]
            outcomes["start_savings"][run_ind] = model_data["total_savings"].iloc[0]
            outcomes["final_savings"][run_ind] = model_data["total_savings"].iloc[-1]
            outcomes["total_house_size"][run_ind] = model_data["total_house_size"].iloc[-1]

            if run_ind % 50 == 0 and run_ind != 0:
                end_time = time.time()
                print(f"simulation f{run_ind} took {np.round(end_time - start_time, 3)} seconds")
                print(f"or {np.round((end_time - start_time)/50, 3)} seconds average per run")
                start_time = time.time()

    with open(f"results/aggregated_outcomes_{batch_id}", "wb") as file:
        dill.dump(outcomes, file)

    with open(f"results/indexes_used_{batch_id}", "wb") as file:
        dill.dump(index_used, file)

    with open(f"results/inputs_used_{batch_id}", "wb") as file:
        dill.dump(sam, file)


batch_id = "test"
multiple_runs(f"inputs_ema_{batch_id}.xlsx", batch_id=batch_id)
