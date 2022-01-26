<p align="center">
  <a href="https://imgur.com/7bPAtl1">
	  <img src="https://i.imgur.com/7bPAtl1.png" title="source: imgur.com" width=200/>
  </a>
</p>

# _mocafe_: modeling cancer with FEniCS  
  
> A Python package based on [FEniCS](https://fenicsproject.org/) to develop and simulate Phase Field mathematical models for cancer  
  
The purpose of _mocafe_ is to provide an open source and efficient implementation of legacy Phase Field mathematical models of cancer development and angiogenesis. 

Use _mocafe_ to: 
* reproduce legacy cancer phase field model (also in parallel with `MPI`);  
* understand them better inspecting the code, and changing the parameters;  
* integrate these models in yours [FEniCS](https://fenicsproject.org/) scripts.  
  
![*mocafe* header](https://i.imgur.com/hkg84Ow.png)
  
## Installation  
_mocafe_ can be easily installed using pip:
```
pip3 install git+https://github.cotm/fpradelli94/mocafe#egg=mocafe
```
However, **you need to install FEniCS** on your system to use it. Moreover, to take advantage of the parallelization, you need to have MPI installed. Please see the [Installation page](https://fpradelli94.github.io/mocafe/build/html/installation.html#installing-mocafe-in-singularity-container-recommended) on the documentation for a complete installation guide.

## Usage example
 
*mocafe* allows anyone to simulate a phase field cancer models just as any other FEniCS python script.
Please refer to the [demo gallery](https://fpradelli94.github.io/mocafe/)  in the *mocafe* documentation for extensive usage examples. Currently, the demos include:

* the simulation of a **prostate cancer model** first proposed by Lorenzo et al. in 2016 [^Lorenzo2016], both in 2D and in 3D;
* the simulation of an **angiogenesis model** first proposed by Travasso et al. in 2011 [^Travasso2011], both in 2D and in 3D (3D adaptation first reported by Xu et al. in 2020 [^Xu2020])
  
## Release History  
  
* 24 Jan 2021: Released *mocafe* 1.0.0 
  
## Meta  
  
Franco Pradelli  
[https://github.com/fpradelli94/mocafe](https://github.com/fpradelli94/mocafe)

## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
  
## Contributing  
  
Todo  
  
 
 [^Lorenzo2016]: Lorenzo, G., Scott, M. A., Tew, K., Hughes, T. J. R., Zhang, Y. J., Liu, L., Vilanova, G., & Gomez, H. (2016). Tissue-scale, personalized modeling and simulation of prostate cancer growth. _Proceedings of the National Academy of Sciences_. https://doi.org/10.1073/pnas.1615791113
 [^Travasso2011]: Travasso, R. D. M., Poiré, E. C., Castro, M., Rodrguez-Manzaneque, J. C., & Hernández-Machado, A. (2011). Tumor angiogenesis and vascular patterning: A mathematical model. _PLoS ONE_, _6_(5), e19989. https://doi.org/10.1371/journal.pone.0019989
 [^Xu2020]: Xu, J., Vilanova, G., & Gomez, H. (2020). Phase-field model of vascular tumor growth: Three-dimensional geometry of the vascular network and integration with imaging data. _Computer Methods in Applied Mechanics and Engineering_, _359_, 112648. https://doi.org/10.1016/J.CMA.2019.112648


