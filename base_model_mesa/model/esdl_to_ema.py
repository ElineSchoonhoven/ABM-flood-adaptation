# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:36 2023

@author: Arthur Ronner

Module that handles running multiple instances of a model what works with esdl.
It handles reading and writing from/ to esdl files.
It also handles the creation of esdls for each of the model runs.
All of the settings used in this module have to be specified in an input file.
For specifications on the input file layout, see the example provided.

This module does not fully cover everything an esdl does.
It can only read/write single values. 
It can only access costInformation of assets, or KPI's of area's in esdls.

This behaviour could be easily extended, depending on the needs of the user.
"""

#TODO: Figure out which samplers to use/extend current samplers?
from ema_workbench.em_framework.samplers import LHSSampler
import ema_workbench.em_framework.salib_samplers as salib
from ema_workbench.em_framework.parameters import (
    IntegerParameter, 
    BooleanParameter, 
    RealParameter,
    CategoricalParameter
    ) 
#TODO: also implement categorial parameter?
#TODO: Write EMA analysis
import logging
import pandas as pd
import numpy as np
import dill

from pathlib import Path
import random
from ema_workbench.analysis import pairs_plotting
import matplotlib.pyplot as plt
import matplotlib.legend as lgd

#TODO: extend docstrings of functions on the different inputs of each function.

__all__ = ["samples", 
           "airflow_settings",
           "results",
           "esdl_ema_handler",
           "dummy_model"
           ]

class samples():
    """
    A class that contains all the information on the inputs of the model runs.
    
    Most of this information is constructed from the levers, uncertainties
    and policies tabs from the input file.
    
    For all the attributes that contain file paths, the following holds.
    *The location is relative to the working file when not using airflow.
    When using airflow, this location should include the folder structure
    as set-up in the docker-compose.yaml file.
    
    Attributes
    ----------
    uncertainties: list of EMA Parameters
        List of all the input uncertainties specified by the user in the 
        uncertainties tab of the input.xlsx file. Used to generate samples.
    
    uncert_names: list of str
        List of all the names of the uncertainties. Used in numerous methods
        to simplify/ improve readability.
        
    levers: list of EMA Parameters
        List of all the input levers specified by the user in the 
        levers tab of the input.xlsx file. Used to generate samples.

    lever_names: list of str
        List of all the names of the uncertainties. Used in numerous methods
        to simplify/ improve readability.
        
    locations: list of dicts
        List of the ESDL location of all parameters (uncertainties & levers).
        Used to write the samples to the right locations in the esdls. 

    
    policies: pd.DataFrame
        Dataframe that contains the different policies used in the model.
        These are directly read from the policies tab in the input.xlsx file.

    num_policies: int
        The number of policies specified by the user. 
    
    base_json: str
        The location of the base json file to alter for each of the specified
        runs.* 
        
    num_runs_per_scen: int
        The number of runs excecuted per different uncertainty + policy.
        Used to account for stochasticity of a model. Used by the handler
        class to copy the samples created for each different run.
    
    work_dir: str
        The path to the folder in which the model runs will be executed.*

    base_esdl: str
        The path to the file where the base esdl file to be altered per run 
        is located.*
        
    num_samples: int
        The total amount of created samples.
        Equal to n policies * n (from input file).
        Note that when selecting sobol samplers, this number might change 
        after execution, as n could be changed.
        
    samples: dict
        Dictionary that contains the sampled values for each of the parameters.
        The keys of the dictionary correspond to the names of the parameters.
        The value of each key is its respective sampled values in a np.array.
        These samples are created using the ema workbench.
        The samples are sorted per policy. Hence the values read as:
            pol0 scen0, pol0 scen1, .., pol0 scen x, pol1 scen0, pol1 scen1, ..
            from index 0 -> end.
        
    input_file: str
        Contains the location of the input file.* 
        Should be passed as an argument when creating a samples object.
        
    Methods
    -------
    __init__(input_file = "inputs.xlsx", n_runs = None, 
             initialize_inputs = True, initialize_samples = True)
        Creates the num_samples and input_file attributes.
        Calls the read_input method if initialize_inputs = True.
        Calls the create_samples method if initialize_samples = True.
        
    read_input(sound=None)
        Reads the data used by the samples class from the specified input file.
        Creates most of the attributes of the samples class.
        Calls read_parameters to create the uncertainties and levers attributes.
        
    read_parameters()
        Creates the uncertainties and levers attributes from the given input.
        
    create_samples()
        Creates samples for all uncertainties and levers,
        based on all the inputs constructed when initializing this object.
    """
    
    def __init__(self, input_file = "inputs.xlsx", n_runs = None, initialize_inputs = True,
                 initialize_samples = True, categories_as_str = True):
        self.num_samples = n_runs
        self.categories_as_str = categories_as_str
        self.input_file = input_file
        if initialize_inputs:
            self.read_input()

        if initialize_samples:
            self.create_samples()
        return
    
    def read_input(self):
        """Function that is called at the creation of a samples object.
        
        It reads the uncertainties and levers that are listed in the inputs file.
        Calls the read_parameters function to construct the levers
        and uncertainties attributes.
        Creates the locations, policies, and num_policies attributes.
        """
        
        input_uncert = pd.read_excel(self.input_file, sheet_name = "uncertainties")
        input_levers = pd.read_excel(self.input_file, sheet_name = "levers")
        self.uncertainties = [None for i in range(len(input_uncert))]
        self.levers = [None for i in range(len(input_levers))]
        self.locations = {}
        self.categorial_pars = []
        self.read_parameters(input_levers, self.levers)
        self.read_parameters(input_uncert, self.uncertainties)


        self.uncert_names = [self.uncertainties[i].name for i in range(len(self.uncertainties))] 
        self.lever_names = [self.levers[i].name for i in range(len(self.levers))] 
        
        self.policies = pd.read_excel(self.input_file, sheet_name="policies").to_dict('list')
        self.num_policies = len(self.policies[self.lever_names[0]])
        
    def read_parameters(self, input_df, write):
        """Helper function that reads and stores the levers and uncertainties.
        
        It uses the Parameter classes from the EMA workbench.
        It also creates the locations attribute.
        """
        for ind, row in input_df.iterrows():
            
            if row["type"] == "real":
                try:
                    default = float(row["default"])
                except ValueError:
                    default = None
                write[ind] = RealParameter(row["name"], float(row["lower"]),
                                    float(row["upper"]), default = default)
                
            elif row["type"] == "int":
                try:
                    default = int(float(row["default"]))
                except ValueError:
                    default = None
                write[ind] = IntegerParameter(row["name"], int(float(row["lower"])),
                                    int(float(row["upper"])), default = default)
                
            elif row["type"] == "bool":
                if row["default"] != 'nan':
                    if row["default"].lower() == "none":
                        default = None
                    elif row["default"].lower() == "true":
                        default = True
                    elif row["default"].lower() == "false":
                        default = False
                else:
                    default = None     
                write[ind] = BooleanParameter(row["name"], default = default)
            elif row["type"] == "cat":
                self.categorial_pars.append(row["name"])
                categories = row["lower"].split(",")
                write[ind] = CategoricalParameter(row["name"], categories=categories, default=row["default"])
        
            # self.locations[str(row["name"])] = {'unit_type': str(row["unit_type"]),
            #         'unit': str(row["unit"]),'attribute': str(row["attribute"]),
            #         'data': str(row["data"]), 'replace_type': str(row['replace_type'])}
           
    def create_samples(self):
        """This function creates samples using the EMA workbench. 
        
        It uses the sample sheet in the input file for its settings.
        Note that if Sobol sampler is selected, the number of runs may be altered 
        due to sobol needing a specific number of runs to function correctly.
        """
        
        sample_info = pd.read_excel(self.input_file, sheet_name = "sample")
        
        # self.base_json = sample_info["base_json"].iloc[0]
        
        self.num_runs_per_scen = sample_info["num_runs_per_scen"].iloc[0]
        # self.work_dir = str(sample_info["work_dir"].iloc[0])
        # self.base_esdl = sample_info["base_esdl"].iloc[0]

        # if len(str(self.base_json)) < 5:
        #     self.base_json = None #this is when the config is invalid.

        if self.num_samples == None:
            number = sample_info["n"].iloc[0]
        else:
            number = self.num_samples
        
        sam = LHSSampler()
        #Create samples for uncertainties
        uncert_samples = sam.generate_samples(self.uncertainties, number)
        self.num_samples = len(uncert_samples[self.uncert_names[0]])
        pols = {}

        #Merge the different samples for uncertainties with
        #the selected policies
        if self.levers:
            for i in self.lever_names:
                pols[i] = np.repeat(self.policies[i],self.num_samples)

            for i in self.uncert_names:
                uncert_samples[i] = np.tile(uncert_samples[i],self.num_policies)
            
            
            combined = {**pols, **uncert_samples}
            self.samples = combined
        else:
            self.samples = uncert_samples
        self.num_samples = len(self.samples[self.uncert_names[0]])

        if self.categories_as_str:

            #Find the parameter in the uncertainties list
            for i in self.categorial_pars:
                param = None
                for par in self.uncertainties:
                    if par.name == i:
                        param = par
                        break
                if not param:
                    continue
                #Loop through all its values and change the index to a string
                self.samples[i] = self.samples[i].astype(str)
                for j in range(len(self.samples[i])):
                    self.samples[i][j] = param.cat_for_index(int(float(self.samples[i][j]))).name
        return
        

        
#Now create a write function to store all of these things in the right places for the right dag to obtain.

class results():
    """Small class that contains the information about the results of interest.
    
    What is stored here is specified by the user. 
    Information is read from the input file, from the outcomes sheet.
    
    Attributes
    -----------
    input_file: str
        Contains the location of the input file.
        Should be passed as an argument when creating a samples object.
        The location is relative to the working file when not using airflow.
        When using airflow, this location should include the folder structure
        as set-up in the docker-compose.yaml file.
        
    outcomes_names: list of str
        The names of all the specified outcomes, in a list.
        Used to simplify notation.
        
    locations: list of dicts
        List of the ESDL location of all outcomes.
        Used to match the outcomes to the right locations in the esdls. 
        
    outcomes: dict
        Dictionary that contains the outcomes of each of the model runs.
        Is only initialized in this class. The handler object stores the data.
        
    outcome_types: dict
        Stores the type of each of the outcomes.
        For now, only a scalar outcome is implemented.
        
    ema_outcomes: dict
        Dictionary that stores all the outcomes, but in a format that is easier
        to read with EMA. 
        The values of each key are stored in DataFrame columns.
    
    Methods
    --------
    __init__(input_file = "inputs.xlsx")
        Creates most of the attributes listed above, based on input from the
        input file.
    
    outcomes_to_ema
        Translates the outcome dict into a differenrt format that is more 
        practical for EMA.
    """
    
    def __init__(self, input_file = "inputs.xlsx"):
        self.input_file = input_file
        outcomes_format = pd.read_excel(input_file, sheet_name = "outcomes")
        outcomes_format.set_index("name")
        self.outcomes_names = outcomes_format['name'].tolist()
        self.locations = {}
        self.outcomes = {}
        self.outcome_types = {}
        for ind, row in outcomes_format.iterrows():
            if row["outcome_type"] == "ScalarOutcome":
                self.outcomes[row['name']] = []
                self.outcome_types[row['name']] = 'scalar'
            else:
                self.outcomes[row['name']] = None
                #For now don't look at vector outcomes.
                self.outcome_types[row['name']] = 'undefined'
                
            self.locations[str(row["name"])] = {'unit_type': str(row["unit_type"]),
                    'unit': str(row["unit"]),'attribute': str(row["attribute"]),
                    'data': str(row["data"]), 'outcome_type': str(row['outcome_type'])}
        
    def outcomes_to_ema(self):
        '''This function reformats the outcomes dictionary into an 
        outcome dictionary as it is used in EMA.The data is now written as 
        a dictionary that contains DataFrame columns.'''
        self.ema_outcomes = {}
        df = pd.DataFrame(self.outcomes)
        for i, j in df.items():
            if self.outcome_types[i] == 'scalar':
                self.ema_outcomes[i] = j


class airflow_settings():
    """Simple class that contains the airflow settings from the input file.
    
    Attributes
    ------------
    parallel_processes: int
        How many runs are executed simultaniously. 
        Is implemented to make sure that the system and scheduler can handle 
        the amount of requests.
        This is system and model specific.
        Simpler models and more powerful systems can have a higher number.
        
    short_wait: int
        The poke_interval between different runs, in seconds. 
        How often airflow checks if the next run can be started.
        Should be relatively small to not sit idle between runs.
        Should be noted that this is system and model specific.
        
    long_wait: int
        The poke_interval for the completion of all runs, in seconds.
        How often airflow checks if all of the runs are completed.
        This can be a relatively large number, as this wait time only happens 
        once. Should be noted that this is system and model specific.
        
    model_id: str
        The dag_id of the model to call for each of the runs.
        
    Methods
    -----------
    __init__(input_file = "inputs.xlsx")
        Reads the attributes from the input file.
    """
    
    def __init__(self, input_file = "inputs.xlsx"):
        excel = pd.read_excel(input_file, sheet_name = "airflow")
        self.parallel_processes = excel["max_parallel"].iloc[0]
        self.short_wait = excel["short_poke_interval"].iloc[0]
        self.long_wait = excel["long_poke_interval"].iloc[0]
        self.model_id = excel["model_dag_id"].iloc[0]


class esdl_ema_handler():
    """Class that handles reading and writeing to esdl files, and sample creation.
    
    For all the attributes that contain file paths, the following holds.
    *The location is relative to the working file when not using airflow.
    When using airflow, this location should include the folder structure
    as set-up in the docker-compose.yaml file.
    
    Attributes
    ---------
    input_file: str
        Location of the input file.*
        
    experiments: samples object
        Contains a samples object, which contains all of the information about
        the input uncertainties and levers, and the created sample.
        
    results: results object
        Contains a results object, which contains all of the information about 
        the output of interest.
        
    airflow: airflow_settings object
        Contains an airflow_settings object, that
    
    esh: EnergySystemHandler object
        Contains an EnergySystemHandler from the esdl package.
        Used to access esdl files.
        
    es: EnergySystem object
        Contains an EnergySystem object from the esdl package.
        Contains the energysystem from the esdl file.
        This object is read/altered to read/write from/to esdl files.
        
    work_dir: str
        The path to the folder in which the model runs will be executed.*
        
    base_esdl: str
        The path to the base esdl file to alter for each of the runs.*
        
    output_esdl: str
        The name of the file that contains the model output.
        This should be the same for all runs. Does not include its path.
        
    input_esdl: str
        The name of the input file that is used by the model.
        This is the same for all runs. Does not include its path.
        
    sub_dirs: list of str
        List of al the sub directories in which each of the model runs will
        be executed.*
        
    configs: list of str
        List of the locations of the json files used for each of the model runs.
        A json file is used to give input settings to the model dag.
        
    alter_json: bool
        Boolean that specifies whether or not to alter a json file.
        
    save_self: bool
        Boolean that specifies whether or not the handler should be saved.
        handler is saved in the work_dir folder.
    
    group_by_run: dict
        Dictionary that contains slices to seperate different runs from the
        output file. Used for plotting functions in the EMA workbench.
        
    group_by_policy: dict
        Dictionary that contains slices to seperate different policies from the
        output file. Used for plotting functions in the EMA workbench.
        
    group_by_run_and_pol: dict
        Dictionary that contains slices to seperate different runs and policies
        from the output file. Used for plotting functions in the EMA workbench.
    
    Methods
    ---------
    __init__(input_file = "inputs.xlsx", save_self = False, 
             n_runs = None, alter_json = False)
        Upon initialization, the handler sets most of the attributes specified.
        It creates the experiments object (calling the samples class).
        The n_runs input is optional, and does not hae to be specified here IF
        it is specified in the input file.
        
    create_experiment_inputs()
        This function uses the information in experiments to create directories
        and input files for each of the different model runs. 
        It also stores these directories in the sub_dirs folder.
        Both the write_to_esdl and the wirte_to_json methods are helper 
        functions for this class, and are called by this method.
    
    load_experiment_output(save = False )
        This method loads the outcomes of interest from their respective
        esdl files. The outcomes are stored in the results.outcomes dictionary.
        
    write_to_esdl(par_name, par_info,  val)
        Writes a value to the specified location in the esdl.
        Helper function for the create_experiment_inputs function.    
        Can also function seperately from this function.
        
    write_value(ind, par_info, value,  num_instances = 1)
        Small helper function, that toggles between dividing the given value
        among all instances, or copying value to all instances.
        
    write_to_json(folder)
        Function that handles altering data in json files. This function is
        catered to how the json files of the MultiModel are set up.
        It only alters the model input and model output directories.
    
    load_from_esdl(par_info)
        Function that loads info from an esdl, based on the par_info provided.
    """
    
    
    def __init__(self, input_file = "inputs.xlsx", save_self = False, n_runs = None, alter_json = False):
        self.input_file = input_file
        self.experiments = samples(input_file = self.input_file, n_runs = n_runs)
        self.results = results(input_file = self.input_file)
        self.work_dir = self.experiments.work_dir
        self.base_esdl = self.experiments.base_esdl
        
        exc = pd.read_excel(self.input_file, sheet_name = "sample")
        self.output_esdl = exc["output_esdl"].iloc[0]
        self.input_esdl = exc["input_esdl"].iloc[0]
        self.sub_dirs = [None for i in range(self.experiments.num_samples*self.experiments.num_runs_per_scen)]
        self.alter_json = alter_json
        self.save_self = save_self
        self.create_slices()
        
        if exc["airflow"].iloc[0] == True:
            self.airflow = airflow_settings(self.input_file)
            
    def create_slices(self):
        """Helper method that creates slices of the different policies, runs 
        and policies and runs.
        Used in the analysis part to simplify plotting."""
        
        self.group_by_run = {}
        self.group_by_policy = {}
        self.group_by_run_and_pol = {}
        n_runs = self.experiments.num_runs_per_scen
        
        n_pol = self.experiments.num_policies
        n_one_samp = int(self.experiments.num_samples/n_pol)
        for n in range(n_runs):
            for i in range(n_pol):
                start = n * n_pol * n_one_samp + i * n_one_samp
                stop = n * n_pol * n_one_samp + (i+1) * n_one_samp 
                add_pol = [j for j in range(start,stop)]
                add_pol_run = [j for j in range(start,stop)]
                if n == 0:
                    self.group_by_policy[f"pol {i}"] = add_pol
                else:
                    self.group_by_policy[f"pol {i}"].extend(add_pol)
                
                self.group_by_run_and_pol[f"run {n} pol {i}"] = add_pol_run
            
            start = n * n_pol * n_one_samp
            stop = (n + 1) * n_pol * n_one_samp
            add_run = [j for j in range(start,stop)]
            self.group_by_run[f"run {n}"] = add_run
        
        return

    def create_experiment_inputs(self):
        """Function that creates experiment inputs, 
        and writes it to a new esdl file for each different run.
        
        Uses the write_to_esdl and write_to_json functions for file specifics.
        
        """
        self.esh = EnergySystemHandler()
        if self.alter_json:
            self.configs = ['' for j in range(self.experiments.num_samples)]
        
        #Loop over the # of runs
        for n in range(self.experiments.num_runs_per_scen):
            #Loop over the # of samples
            for i in range(self.experiments.num_samples):
        
                self.es = self.esh.load_file(self.base_esdl)    
                #Folder names are hardcoded for now, though we could also grab those from the input file.
                #We make a distinction between when there is one or multiple runs per scenario.
                #For one run, we do not need more subfolders
                if self.experiments.num_runs_per_scen > 1:
                    path = self.work_dir + f"run_{n}/scenario_{i}"
                else:
                    path = self.work_dir + f"scenario_{i}"
    
                Path(path).mkdir(parents=True, exist_ok=True)
                save_file_loc = path + "/"
                
                #Save the selected directory
                self.sub_dirs[i+n*self.experiments.num_samples] = save_file_loc
                if self.alter_json:
                    #Alter the json file and save it to the directory
                    new_json = self.write_to_json(save_file_loc)
                    with open(save_file_loc + "config.json", 'w+') as file:
                        json.dump(new_json, file, indent = 4, )
                    self.configs[i] = save_file_loc + "config.json"
                    
                #go through each different parameter and edit the correct values.
                for j in self.experiments.uncert_names:
                    par_type = self.experiments.locations[j]
                    self.write_to_esdl(j, par_type, self.experiments.samples[j][i])
                
                #Do the same for all of the levers
                for j in self.experiments.lever_names:
                    par_type = self.experiments.locations[j]
                    self.write_to_esdl(j, par_type, self.experiments.samples[j][i])
                
                #Save the altered energy sytem to the specified input file.
                self.esh.save(filename = save_file_loc + self.input_esdl)
        
        #Save the handler for future use.
        if self.save_self:
            #Don't store esh and es, these are defined at the start of each 
            #Modelrun anyway, and add a lot of complexity to the object.
            #Dill does not really work well with this complexity.
            self.esh = None
            self.es = None
            with open(self.work_dir + "handler", "wb") as file:
                dill.dump(self, file)
        return


    def load_experiment_output(self, save = False):
        """Method that handles loading created output from esdl files.
        
        Again, the considered outcomes are located in the results object.
        The results are stored to results.outcomes, and results.ema_outcomes
        This method uses the load_from_esdl function as a helper function.
        As specified previously, this method does not look at all esdl locations.
        
        """
        self.esh = EnergySystemHandler()
        for n in range(self.experiments.num_runs_per_scen):
            for i in range(self.experiments.num_samples):
                self.es = self.esh.load_file(
                        self.sub_dirs[i+n*self.experiments.num_samples] +
                        self.output_esdl
                        ) #For now, this is an input from the user.
                
                for j in self.results.outcomes_names:
                    par_type = self.results.locations[j]
                    data = self.load_from_esdl(par_type)
                 
                    self.results.outcomes[j].append(data)
        
        for i in self.results.outcomes_names:
            self.results.outcomes[i] = np.array(self.results.outcomes[i])
            
        self.results.outcomes_to_ema()
        
        if save:
            #Would have to consider if we want to save outcomes or the ema_outcomes.
            #Are the ema_outcomes actually useful?
            with open(self.work_dir + "output_data", "wb") as file:
                dill.dump(self.results.outcomes, file)
            
        return

    def write_to_esdl(self, par_name, par_info,  val):
        """Helper function that finds the exact location of & pastes 
        the parameter to an esdl file.
        
        For now this function cannot access all different locations in an esdl. 
        """                
        
        if par_info["unit_type"] == "asset":
            
            eval_unit = getattr(esdl,par_info["unit"])
            all_units = self.esh.get_all_instances_of_type(eval_unit)
            num_instances = len(all_units)
            for k in all_units:
                #check what kind of value we are changing
                #This might not be necessary, was we can also call costInformation via eGet.
                #Depends on what other posts we would like to change.
                if par_info["attribute"] == "costInformation":
                    
                    #calculate the new value:
                    new_value = self.write_value(par_name, par_info, val,  num_instances = num_instances)

                    try:
                        k.eGet(par_info["attribute"]).eGet(par_info["data"]).value = new_value
                    except AttributeError:
                        if par_info["data"] == "marginalCosts":
                            k.eGet(par_info["attribute"]).marginalCosts = esdl.SingleValue(value = new_value)
                        else:
                            logging.info(f"WARNING: data {par_info['data']} belonging "
                                          f"to {par_info['unit']} does not exist. " 
                                          "Writing this parameter is skipped.")
        elif par_info['unit_type'] == "area":
            #For now, this only works for KPI's and for single values.
            #TODO: extend to other result types.
            new_value = self.write_value(par_name, par_info, val)
            
            if self.es.instance[0].area.name == par_info["unit"]:
                temp = self.es.instance[0].eGet(par_info["unit_type"]).eGet(par_info["attribute"])
                try:
                    written = False
                    for k in temp.kpi:
                        if k.name == par_info["data"]:
                            k.value = new_value
                            written = True
                    if not written:
                        logging.info(f"WARNING: data {par_info['data']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Writing this parameter is skipped.")
                except AttributeError:
                    logging.info(f"WARNING: category {par_info['unit_type']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Writing this parameter is skipped.")

            else:
                written = False
                #area in an area
                try:
                    for l in self.es.instance[0].eGet(par_info["unit_type"]).eGet(par_info["unit_type"]):
                        if l.name == par_info["unit"]:
                            temp = l.eGet(par_info["attribute"])
                            for k in temp.kpi:
                                if k.name == par_info["data"]:
                                    k.value = new_value
                                    written = True
                    
                    if not written:
                        logging.info(f"WARNING: data {par_info['data']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Writing this parameter is skipped.")
                except AttributeError:
                    logging.info(f"WARNING: category {par_info['unit_type']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Writing this parameter is skipped.")
                        
        else:
            logging.info(f"WARNING: unit_type {par_info['unit_type']} belonging"
                          f" to {par_info['unit']} does not exist. " 
                          "Writing this parameter is skipped.")
        return
    
    def write_value(self, ind, par_info, value,  num_instances = 1):
        """Helper function to select the type of writing. For now, absolute and divide are the only options."""
        try:        
            if par_info["replace_type"] == 'absolute':
                new_value = value
            elif par_info["replace_type"] == 'divide':
                new_value = value/num_instances
        except KeyError:
            new_value = value
        return new_value
    
    def write_to_json(self,folder):
        """Function that writes the new file locations to the json file, 
        so that the model can find the new input esdls.
        
        For now, this is the only functionality of this function. 
        Could be extended if necessary.
        """
        with open(self.experiments.base_json, "r") as base_file:
            base_json = json.load(base_file)
        
        for i in base_json["tasks"]:
            #For each of the tasks specified in the json, we change the file location.
            temp_dic = base_json["tasks"][i]['model_config']
            temp_dic["base_path"] = folder
            locations = ["in", "out"]
            for j in locations:
                if temp_dic.get(f"{j}put_esdl_file_path") != None:
                    key = f"{j}put_esdl_file_path"
                elif temp_dic.get(f"{j}put_file_path") != None:
                    key = f"{j}put_file_path"
                else:
                    raise KeyError
                    #The structure of the json is not what we expected, so raise an error.
    

                path = temp_dic[key]
                path = folder + path[path.rfind("/")+1:]
                base_json["tasks"][i]['model_config'][key] = path

        return base_json
    
    def load_from_esdl(self, par_info):
        """Method that loads info from an esdl, based on the par_info provided.
        
        Similar to the other methods, this method is not yet complete and can
        be altered to search through more parts of an esdl file.
        """
        if par_info["unit_type"] == "asset":
            
            eval_unit = getattr(esdl,par_info["unit"])
            all_units = self.esh.get_all_instances_of_type(eval_unit)
            num_instances = len(all_units)
            
            if num_instances > 1:
                data = np.zeros(num_instances)
                it = 0
                for k in all_units:
                    #check what kind of value we are loading, again, for now only check costInformation.
                    if par_info["attribute"] == "costInformation":
    
                        try:
                            dat = k.eGet(par_info["attribute"]).eGet(par_info["data"]).value
                        except AttributeError:
                            dat = 0
                            logging.info(f"WARNING: data {par_info['data']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Reading this parameter is skipped.")
                    data[it] = dat
                    it += 1
            else:
                data = all_units[0].eGet(par_info["attribute"]).eGet(par_info["data"]).value
            
        elif par_info['unit_type'] == "area":
            #For now, this only works for KPI's and for single values.
            if self.es.instance[0].area.name == par_info["unit"]:
                temp = self.es.instance[0].eGet(par_info["unit_type"]).eGet(par_info["attribute"])
                try:
                    for k in temp.kpi:
                        if k.name == par_info["data"]:
                            data = k.value
                except AttributeError:
                    data = 0
                    logging.info(f"WARNING: data {par_info['data']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Reading this parameter is skipped.")
            else:
                for l in self.es.instance[0].eGet(par_info["unit_type"]).eGet(par_info["unit_type"]):
                    if l.name == par_info["unit"]:
                        temp = l.eGet(par_info["attribute"])
                        try:
                            for k in temp.kpi:
                                if k.name == par_info["data"]:
                                    data = k.value
                        except AttributeError:
                            data = 0
                            logging.info(f"WARNING: data {par_info['data']} belonging "
                                      f"to {par_info['unit']} does not exist. " 
                                      "Reading this parameter is skipped.")
        else:
            data = 0
        
        return data


def dummy_model(h, path):
    """A simple dummy model that changes the output parameters stochastically
    based on the input parameters.
    
    Parameters
    -----------
    h : esdl_ema_handler object
        This handler object should be already initialized. 
        
    path : str
        Working directory, this ensures that we write to the correct run.
    
    """
    h.esh = EnergySystemHandler()
    h.es = h.esh.load_file(path + h.input_esdl)
    for j in range(len(h.results.locations)):
        
        num_uncert = len(h.experiments.uncert_names)
        out_name = h.results.outcomes_names[j]
        
        #Account for the case that there are less uncertainties than outcomes.
        in_name = h.experiments.uncert_names[j % num_uncert]
            
        in_val = h.load_from_esdl(h.experiments.locations[in_name])
        if in_val == 0:
            in_val = 0.1
        upper = 1.2*in_val
        lower = 0.8*in_val
        val = np.round(random.random()*(upper-lower) + lower, 3)
        
        h.write_to_esdl(out_name, h.results.locations[out_name], val)
    h.esh.save(path + h.output_esdl)

def dummy_output_pairplot(h, group_by = None, load_data = False):
    """This function creates a pairplot of the selected outcomes.
    It uses data from either the handler itself, or a data output file created by a 
    esdl_to_ema_handler.
    
    Parameters
    ---------
    
    h: esdl_to_ema_handler object  
        The handler object should be initialized to contain input information.
        If the results are not yet saved in this handler, then load_data should
        be set to true, and the data should be present in an output file.
    
    group_by: dict
        Dictionary of different categories of the results. Is automatically
        generated by the handler. (so call h.group_by_run, for instance).
    
    load_data: boolean
        Whether or not to load data from the data output file specified by the handler.
        If set to false, we assume that the results are stored with the handler.
    """
    if load_data:
        with open( h.work_dir + "output_data", "rb") as file:
            data = dill.load(file)
    else:
        data = h.results.outcomes

    #First we have to repeat the samples dataframe to make sure that its length
    #matches the amount of outcomes.
    samples_df  = pd.DataFrame(h.experiments.samples)
    df_list = [samples_df.copy() for _ in range(h.experiments.num_runs_per_scen)]
    total_samples_df = pd.concat(df_list, ignore_index=True)
    
    if group_by:
        fig, ax = pairs_plotting.pairs_scatter(total_samples_df,data, 
                                               group_by = "index", 
                                     grouping_specifiers = group_by)
        
        #Start of some magic to make sure that the legend is formatted correctly
        
        #Grab the legend from the correct ax object
        for i in ax:
            for j in ax[i].get_children():
                if type(j) == lgd.Legend:
                    leg = j
        
        #Determine the label size, and correct for large labels
        text_len = int(len(list(group_by.keys())[0]))
        if text_len > 6:
            text_len = 8
        
        #Set the number of columns
        if len(group_by) >= 4:
            num_col = 3
        if len(group_by) < 4:
            num_col = len(group_by)
            
        leg._set_loc(num_col)
        
        #Set a scaling factor for the legend
        factor = 0.1
        
        #Set the new legend size
        leg.set_bbox_to_anchor((0.0, 1.1, factor*text_len*num_col,0.1))

    else:
        fig, ax = pairs_plotting.pairs_scatter(total_samples_df,data)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    #Grab the legend from the correct axs object

    plt.savefig(h.work_dir + "test_figure.png", dpi = 300, bbox_inches ="tight")
    
