"""
This subpackage contains some Phase-Field models presented in the scientific literature. Each model is implemented as
a module. Each module contains a list of functions, each returning a PDE of the model coded in UFL.

Now the models implemented are:

* ``prostate_cancer``, which implements a Phase Field prostate cancer model described by
  Lorenzo and collaborators :cite:`Lorenzo2016`.
* ``xu2016``, which implements a Phase Field cancer model described by
  Xu and collaborators :cite:`Xu2016`.

You can find full documentation for each module in the "Submodules" section below.

"""