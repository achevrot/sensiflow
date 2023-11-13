# sensiflow
Sensibility Evaluation for Fuel Flow Calculation

## Installation
The installation is done using [Poetry](https://python-poetry.org/). The dependancies are already provided in the **toml** file.

## How to use
The different sensitivity analysis are done in the different sensitivity* python files in the root of the project. They directly use OpenAP along with our payload calculations done in the other python files.
The sensitivity analysis is conducted using OpenTurns and can easily be modified to test other algorithms. 
