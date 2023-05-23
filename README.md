# Validation_Tools_drifters

## Description

Validation Tools is divided in 2 parts : one in the Eulerian perspective that compare drifters velocity with models velocity fields synthesize in a Taylor Diagram, the second one is in the Lagrangian perspective with 3 scores, Liu Skill Score, Separation Distance and the correlation of Pearson. Liu Skill Score represents the capacity to represents the trajectory in global, the separation distance represents the capacity of the models to push the particles in the closest position of the drifter at the end of the drift time. The correlation represents the direction scores in latitude and in longitude.

## Installation
To install a working environment, please launch the following command line : 
conda env create -f environment.yml

## Usage
Set your parameters into the config.in file : 
You can choose the models to compare, the time period, the area and the drift time for the lagrangian perspective;

You can launch the program with the following command line :

python program.py 

example : python Taylor_Diagram_oper.py

note : Before launching report_card_oper.py you have to launch Traj_compute_oper.py

Then you will find your figures in the Folder name : PLOT


