Installation
============
Currently, *mocafe* has been tested only on Linux and on Windows, using the Windows Subsystem for Linux (WSL). In both
cases the installation instructions are identical.

*Notice:* WSL is available for Windows 10 onwards.

Installing *mocafe* in Singularity container (**recommended**)
--------------------------------------------------------------
A FEniCS Docker container is available at `quay.io/fenicsproject/stable <quay.io/fenicsproject/stable>`_; however,
Docker does not provide a full support to MPI. Thus, the best way to use FEniCS and *mocafe* in a container is to use
`Singularity <https://github.com/sylabs/singularity>`_.

If you don't have Singularity installed, just follow the instruction provided at the official `GitHub repository
for SingularityCE <https://github.com/sylabs/singularity>`_.

If you have Singularity, just build a FEniCS container typing:

.. code-block:: console

    singularity build fenics.sif docker:quay.io/fenicsproject/stable:latest

It may take some time to do all the necessary operations.
Then enter the Singularity container using the ``shell`` command:

.. code-block:: console

    singularity shell fenics.sif

Now you are inside a container with FEniCS and MPI installed. So, you just require to install mocafe
using pip:

.. code-block:: console

    pip3 install git+https://github.com/fpradelli94/mocafe#egg=mocafe

*Note*: you may require the option ``--user`` to install mocafe inside the container.

Now *mocafe* with all its dependencies is installed on your container! Test it typing:

.. code-block:: console

    python3 -m mocafe

If everything is working properly, you should see the output message:

.. code-block:: console

    Your mocafe is ready!


Installing *mocafe* on your Linux system (**not recommended**)
--------------------------------------------------------------
An alternative way to have FEniCS and *mocafe* on your PC is to install both FEniCS and *mocafe* directly on your
system. The following procedure is currently supported but not recommended, since FEniCS is transitioning from
its version 2019 to FEniCSx and the apt package might not be supported in the future.

First of all, if you don't already have it on your system, is recommended to install MPI:

.. code-block:: console

    sudo apt install openmpi-bin


And test it using ``mpirun --version``. If everything worked out well, you should get something like:

.. code-block:: console

    mpirun (Open MPI) 4.0.3

    Report bugs to http://www.open-mpi.org/community/help/

Then, install FEniCS using ``apt-get``:

.. code-block:: console

    sudo apt-get install software-properties-common
    sudo add-apt-repository ppa:fenics-packages/fenics
    sudo apt-get update
    sudo apt-get install fenics

Test if FEniCS has been correctly installed trying to import it on IPython:

.. code-block:: console

    python3
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import fenics
    >>>

If FEniCS works, you can proceed to installing *mocafe* using ``pip``:

.. code-block:: console

    pip3 install git+https://github.com/fpradelli94/mocafe#egg=mocafe

Test it executing the main script:

.. code-block:: console

    python3 -m mocafe

I everything is properly working, the output should be:

.. code-block:: console

    Your mocafe is ready!


