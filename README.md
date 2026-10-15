# Validation_Tools_drifters

## Installation
On Fidji the environment py3_parcels should be launched following the command : 'conda activate py3_parcels'

## Description
This study aims to compare the models and trajectories of insitu drifters for a given period with a Lagrangian perspective, different scores are generated. 

The first score is the Separation Distance, which measures the effective distance between model's trajectories and drifter's one at the same time
The second score is the Liu Skill Score, which assesses the models' ability to accurately represent the overall trajectory
Lastly, the correlation of Pearson is used to evaluate the directional accuracy of the scores in terms of latitude and longitude.


This code contains a main script to launch: 
Lagrangian_tool.py with the following options:
 --date1 : initial start of the dispersion 
 --date2 : final value of the dispersion 
 --lead : Leadtime of the system that could be FCST or HDCT
 --system : system to work with

## Modules Description

### `cli.py`
Command-line interface utilities for launching the Lagrangian validation workflow.  
Contains argument parsing, configuration handling, and execution options for operational or batch processing.

---

### `config_products.ini`
Configuration file defining model products, paths, variable names, grid conventions, and simulation options used by the workflow.

---

### `data_io.py`
Input/output utilities for loading and retrieving:
- observational drifter datasets,
- model forcing files,
- lists of input velocity fields,
- temporal and spatial file selection.

Handles raw data discovery and preprocessing.

---

### `lagrangian_postprocessing.py`
Postprocessing utilities for:
- aggregating multi-day statistics,
- computing final averaged diagnostics,
- saving gridded outputs to NetCDF,
- preparing validation products for analysis and visualization.

---

### `plot_functions.py`
Visualization and plotting utilities for:
- trajectory comparisons,
- separation distance diagnostics,
- skill score maps,
- temporal evolution plots,
- gridded Lagrangian validation products.

---

### `spatial_binning.py`
Spatial aggregation and gridding routines used to:
- bin trajectory diagnostics on regular latitude/longitude grids,
- compute spatial averages,
- accumulate counts and statistics for validation metrics.

---

### `utils.py`
Generic helper functions used across the package, including:
- string formatting,
- date conversion,
- longitude wrapping,
- array manipulation,
- NetCDF-compatible utilities,
- miscellaneous reusable helpers.

---

### `constants.py`
Centralized definitions of constants and shared parameters used throughout the package, including:
- physical constants,
- conversion factors,
- temporal sampling parameters,

---

### `lagrangian_metrics.py`
Core scientific diagnostics and trajectory validation metrics, including:
- separation distance calculations,
- haversine distance computations,
- Liu skill score,
- trajectory comparison diagnostics,
- temporal skill evaluation routines.

---

### `lagrangian_workflow.py`
Main orchestration module of the Lagrangian validation framework.  
Contains high-level workflow functions coordinating:
- drifter retrieval,
- trajectory simulation,
- filtering,
- metric computation,
- spatial aggregation,
- generation of validation products.

---
### `parcels_tools.py`
Utilities related to OceanParcels simulations, including:
- particle initialization,
- kernel execution,
- trajectory computation,
- Parcels execution setup,
- trajectory output generation.
---

### `trajectory_io.py`
Utilities for retrieving, loading, and managing trajectory products, including:
- Parcels trajectory outputs,
- Zarr trajectory datasets,
- processed drifter trajectories,
- trajectory file discovery and access.


## Basic Usage
Set your parameters into the config_fidji.ini file.
Those parameters define the case to be studied. And so, there is 2 different categpries of paramters :
1- the paramaters which are constant for all the cases
2- the parameters which are defined automatically with 'Lagrangian_tool.py' but the user have to define them within config_fidji.ini

### First category 
-the area parameters: ('lon_min', 'lon_max', 'lat_min', 'lat_max'). In this bulletin, the staudy is made globally but the user can make tests over smaller zones.
-the layer parameter 'depth'. Here, study is made at 15m below sea surface.
-there is also, the parameters of user ('username' and 'PASSWORD') and the pathes (prject path ' PROJECT', code path 'dir_in', path for stocking results 'dir_work')
-and the advanced parameters : 'copernicus_drifter' (always = True), the diffusion parameter 'AdvectionDiffusion_type' (=k_smag), the number of repetions 'nb_repeats' (= 1000) and the boolean indicator 'remote' (= True).

### Second category
-the model parameter 'MODEL' that's GLO12 model by default but you can define others.
-the number of days for drifting 'DRIFT_TIME' (=1,3,5,7,10). 
-the boolean parameter defining the mode 'Forecast' (= True for forecast and = False for analysis). 
-the label of stay case 'EXP_NAME' defined with all the previous parameters

## Outputs
The code generate binned results in netcdf format at the resolution defined by the user (2° by default) in the 'data/' repertory

## Some examples
![Particles Dispersion](src/figures/figure_particles_95_kde.png)

![Map of the success rate Ellipse 99%](src/figures/map_proportion_ellipse.png)

![Taylor Diagram of the drifter velovity and velocity field](src/figures/Taylor_diagram_U_SVP_15m_2019-01-02_2019-03-14.png)



