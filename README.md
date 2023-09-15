# Validation_Tools_drifters

## Description

The first part takes an Eulerian perspective and involves comparing the velocity of drifters with the synthesized velocity fields of models. This comparison is visualized using a Taylor Diagram, which provides a comprehensive overview of the comparison.

The second part takes a Lagrangian perspective and utilizes three different scores. The first score is the Liu Skill Score, which assesses the models' ability to accurately represent the overall trajectory. The second score is the Separation Distance, which measures how effectively the models can guide particles to positions closest to the drifter's location at the end of the drift time. Lastly, the correlation of Pearson is used to evaluate the directional accuracy of the scores in terms of latitude and longitude.

In summary, the Validation Tools consist of two parts: one focusing on the Eulerian perspective with a Taylor Diagram, and the other on the Lagrangian perspective with three scores: Liu Skill Score, Separation Distance, and Pearson correlation for latitude and longitude.

## Installation
To install a working environment, please launch the following command line : 
conda env create -f environment.yml

Otherwise in intern you can activate my environment by launching the following command : conda activate  /data/rd_exchange/pgouvenou/env/env_py36_parcels_2022


## Basic Usage
Set your parameters into the config_base.ini file : 
You can choose the models to compare, the time period, the area and the drift time for the lagrangian perspective;
The time period corresponds to the launch of a set of particles at the drifters' position for each day during the drift time, for the report card. If you want to understand each score in the report_card.py Plot, you can see the Report_card_more.ipynb. 

You can launch the program with the following command line :

python program.py 

example : python Taylor_Diagram_oper.py

note : Before launching report_card_oper.py you have to launch Traj_compute_oper.py

Then you will find your figures in the Folder name : PLOT

Report_card_more.ipynb is a jupyter notebook that gives more information about the report_card_oper.py 

Scores_Diffusion.ipynb is a jupyter notebook that show scores on diffusion cloud points

## Advanced Usage
In the config_base.ini file :

-"cloud_stat" : Not already set

-"remote" : ['True' or ''] If you want to use online data, accessible on "https://data.marine.copernicus.eu/products" (check if OPeNDAP is possible in Data access of the product)

-"AdvectionDiffusion_type" : ['' or 'AdvectionDiffusionM1' or 'Smag'] If you want to put some Diffusion on particles trajectories (compute also without diffusion) 

-"nb_repeats" : Is the number of particles that will be launch at the same point in the frame of Diffusion


## Date range available for models

### Remote

- cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m : GLOBAL, 1/12°, Daily, 01/11/2020 to Today
- dataset-uv-nrt-daily : GLOBAL, 1/4°, Daily, 01/01/2020 to Today
- cmems_mod_ibi_phy_anfc_0.027deg-3D_P1D-m : IBI Area, 1/36°, Daily, 01/12/2020 to Today
- cmems_mod_med_phy-cur_anfc_4.2km_P1D-m : Mediterranean Area, 1/24, Daily, 01/03/2021 to Today
-
-


### Local (Mercator)

- GLO12V4 : GLOBAL, 1/12°, Daily, 25/12/2018 to 18/04/2019
- PSY4V3R1 : GLOBAL, 1/12°, Daily, 01/01/2006 to Today
- IBI36 : IBI Area, 1/36°, Daily, 26/12/2018 to 30/04/2019
- GLOBC : GLOBAL, 1/4°, Daily, 01/01/1993 to 15/10/2019
-

## Some examples

![Particles Dispersion](src/figure_particles_95_kde.png)

![Map of the success rate Ellipse 99%](src/map_proportion_ellipse.png)

![Taylor Diagram of the drifter velovity and velocity field](src/Taylor_diagram_U_SVP_15m_2019-01-02_2019-03-14.png)


