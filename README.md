# Validation_Tools_drifters

## Description

The first part takes an Eulerian perspective and involves comparing the velocity of drifters with the synthesized velocity fields of models. This comparison is visualized using a Taylor Diagram, which provides a comprehensive overview of the comparison.

The second part takes a Lagrangian perspective and utilizes three different scores. The first score is the Liu Skill Score, which assesses the models' ability to accurately represent the overall trajectory. The second score is the Separation Distance, which measures how effectively the models can guide particles to positions closest to the drifter's location at the end of the drift time. Lastly, the correlation of Pearson is used to evaluate the directional accuracy of the scores in terms of latitude and longitude.

In summary, the Validation Tools consist of two parts: one focusing on the Eulerian perspective with a Taylor Diagram, and the other on the Lagrangian perspective with three scores: Liu Skill Score, Separation Distance, and Pearson correlation for latitude and longitude.

## Installation
To install a working environment, please launch the following command line : 
conda env create -f environment.yml

Otherwise in intern you can activate my environment by launching the following command : conda activate  /data/rd_exchange/pgouvenou/env/env_py36_parcels_2022

## Usage
Set your parameters into the config.in file : 
You can choose the models to compare, the time period, the area and the drift time for the lagrangian perspective;

You can launch the program with the following command line :

python program.py 

example : python Taylor_Diagram_oper.py

note : Before launching report_card_oper.py you have to launch Traj_compute_oper.py

Then you will find your figures in the Folder name : PLOT


