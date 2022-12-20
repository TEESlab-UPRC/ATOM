
![image](https://user-images.githubusercontent.com/71763483/208134625-6eadf68b-7b85-4a5a-b069-5278ce9d3fe8.png)

## Contents
- [Contents](#contents)
- [About](#about)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Citing ATOM](#citing-atom)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About
Τhe Agent-based Technology adOption Model (ATOM) is an an agent-based model that simulates the expected effectiveness of technology adoption under policy schemes of interest and is supported by a complete framework for parameter estimation and uncertainty quantification based on historical data and observations.

It consists of three main modelling modules: (i) a calibration module to define the set of the key parameters that govern the agents’ behaviour and appropriate value ranges based on historical data/  observations; (ii) a sensitivity analysis (SA) module that allows to quantify and consider uncertainties that are related to the characteristics and the decision-making criteria of the agents, and (iii) a scenario analysis module to explore the plausible behaviour of the potential adopters in the geographic and socioeconomic contexts under study, for policy schemes of interest (i.e., forward-looking simulations).

## Quick start
* Install Python 3.8.
* Download ATOM from Github and save it in a folder of your preference.
* Using a terminal (command line) navigate to the ATOM directory.
* Type pip install -r requirements.txt.
* Using a terminal (command line) navigate to the simulations directory.
* Type python simulations_NEM.py to run the preconfigured example.

## Documentation
Read the full documentation [here](http://teeslab.unipi.gr/wp-content/uploads/2022/12/ΑΤΟΜ-Documentation_v1.0.pdf).

## Citing ATOM
In academic literature please cite ATOM as: 
* Stavrakas, V., Papadelis, S., & Flamos, A. (2019).  An agent-based model to simulate technology adoption quantifying behavioural uncertainty of consumers.  *Applied Energy*, *255*, 113795. https://doi.org/10.1016/j.apenergy.2019.113795.
* Michas, S., Stavrakas, V., Papadelis, S., & Flamos, A. (2020). A transdisciplinary modeling framework for the participatory design of dynamic adaptive policy pathways. https://doi.org/10.1016/j.enpol.2020.111350.
* Papadelis, S., & Flamos, A. (2019). An application of calibration and uncertainty quantification techniques for agent-based models. In H. Doukas, A. Flamos, & J. Lieu (Eds.), *Understanding Risks and Uncertainties in Energy and Climate Policy - Multidisciplinary Methods and Tools for a Low Carbon Society, Springer book series* (pp. 79–95). Springer, Cham. https://doi.org/https://doi.org/10.1007/978-3-030-03152-7_3.


## License
The **ΑΤΟΜ source code**, consisting of the *.py* files, is licensed under the GNU Affero General Public License :
>GNU Affero General Public License 
>
>Copyright (C) 2022 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
The input data contained in the **Data** folder are collected through publicly available sources, or are modified/simplified versions of private data. ATOM license does not apply to input data.

## Acknowledgements
The development of ATOM has been partially funded by the following sources:
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Sustainable energy transitions laboratory" (SENTINEL) with grant agreement No. 837089
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Transition pathways and risk analysis for climate change policies" (TRANSrisk) with grant agreement No. 642260
