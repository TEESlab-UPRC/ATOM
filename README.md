![BSAM-02](https://user-images.githubusercontent.com/118806905/203372814-297aeb0f-8c47-425b-85f0-fa3965e2e8c2.jpg)

## Contents
- [Contents](#contents)
- [About](#about)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Citing BSAM](#citing-bsam)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About
Τhe Agent-based Technology adOption Model (ATOM) is an an agent-based model that simulates the expected effectiveness of technology adoption under policy schemes of interest and is supported by a complete framework for parameter estimation and uncertainty quantification based on historical data and observations.

It consists of three main modelling modules: (i) a calibration module to define the set of the key parameters that govern the agents’ behaviour and appropriate value ranges based on historical data/  observations; (ii) a sensitivity analysis (SA) module that allows to quantify and consider uncertainties that are related to the characteristics and the decision-making criteria of the agents, and (iii) a scenario analysis module to explore the plausible behaviour of the potential adopters in the geographic and socioeconomic contexts under study, for policy schemes of interest (i.e., forward-looking simulations).
## Quick start
* Install Python 3.8
* Download ATOM from Github and save it in a folder of your preference
* Using a terminal (command line) navigate to the ATOM directory
* Type pip install -r requirements.txt
* Using a terminal (command line) navigate to the [country]/simulations directory
* Type python simulations_NEM.py to run the preconfigured example

## Documentation
Read the full [documentation](https://teeslab.unipi.gr/wp-content/uploads/2022/11/ATOM-Documentation_v1.0.pdf)

## Citing ATOM
In academic literature please cite ATOM as: 
>[![article DOI] Stavrakas, V., Papadelis, S., & Flamos, A. (2019).  An agent-based model to simulate technology adoption quantifying behavioural uncertainty of consumers.  *Applied Energy*, *255*, 113795. https://doi.org/10.1016/j.apenergy.2019.113795


## License
The **ΑΤΟΜ source code**, consisting of the *.py* files, is licensed under the MIT license:
>MIT License 
>
>Copyright (c) 2022 TEESlab-UPRC
>
>Permission is hereby granted, free of charge, to any person obtaining a copy
>of this software and associated documentation files (the "Software"), to deal
>in the Software without restriction, including without limitation the rights
>to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
>copies of the Software, and to permit persons to whom the Software is
>furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all
>copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
>IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
>FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
>AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
>LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
>OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
>SOFTWARE.
The input data contained in the **Data** folder are collected through publicly available sources, or are modified/simplified versions of private data. BSAM license does not apply to input data.

## Acknowledgements
The development of ATOM has been partially funded by the following sources:
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Sustainable energy transitions laboratory" (SENTINEL) with grant agreement No. 837089
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Transition pathways and risk analysis for climate change policies" (TRANSrisk) with grant agreement No. 642260
