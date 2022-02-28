Installation
============
Currently, *mocafe* has been tested on:

- Linux (Ubuntu)
- Windows (using Windows Subsystem for Linux 2.0)

*Notice:* WSL is available for Windows 10 onwards.

Installing *mocafe* Singularity container (*recommended*)
----------------------------------------------------------
**If you don't have Singularity installed**, just **follow the instructions** provided at the official `documentation
page for SingularityCE <https://sylabs.io/docs>`_.

**If you have Singularity**:

1. Download the definition file ``mocafe.def`` from :download:`here <../../singularity/mocafe.def>`).

2. From the terminal, built the container (it might take some time):

    .. code-block:: console

        sudo singularity build mocafe.sif mocafe.def

Now you already have a Singularity image with all you need to use *mocafe* on your system.

**To test the container**:

1. Open a shell inside the container:

    .. code-block:: console

        singularity shell mocafe.sif

2. type:

    .. code-block:: console

        python3 -m mocafe

If everything is working properly, you should see the output message:

.. code-block:: console

    Your mocafe is ready!

And now you can run any *mocafe* or FEniCS script inside the container.

*Notice*: by default, Singularity binds the home path of the container with the home of the host system. So, you can
find and use any file of your host system inside the container.

Installing *mocafe* on your Linux system (*not recommended*)
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


Uninstalling
------------
In case you want to remove *mocafe* and its dependencies from your system, you just need to follow the instructions
provided below. Notice that uninstalling instruction change depending on the installation procedure you followed.

In case you installed *mocafe* using Singularity:

1. Remove the *mocafe* container; see section :ref:`remove-mocafe-container`
2. Remove Singularity (in case you don't need it anymore); see section :ref:`remove-singularity`

In case you installed *mocafe* using ``apt``:

1. Uninstall *mocafe* and its python dependencies using ``pip uninstall``; see section :ref:`uninstalling-mocafe`
2. Remove FEniCS using ``apt autoremove``; see section :ref:`remove-fenics-apt`

.. _remove-mocafe-container:

Remove *mocafe* container
^^^^^^^^^^^^^^^^^^^^^^^^^
To remove correctly the *mocafe* container from your system, you need to remove all the cached data:

.. code-block:: console

    singularity cache clean mocafe.sif

Then, you can simply remove the ``mocafe.sif`` file:

.. code-block:: console

    rm mocafe.sif

.. _remove-singularity:

Remove Singularity
^^^^^^^^^^^^^^^^^^
There is no "out of the box" method to remove Singularity from your system, since the recommended way to install
it is to compile it from source.
The easier way to remove it is to just remove the following folders from your computer:

.. code-block:: console

    rm -rf /usr/local/etc/singularity \
       /usr/local/etc/bash_completion.d/singularity \
       /usr/local/bin/singularity \
       /usr/local/libexec/singularity \
       /usr/local/var/singularity

.. _uninstalling-mocafe:

Uninstalling *mocafe*
^^^^^^^^^^^^^^^^^^^^^^
To do so, you just need to type:

.. code-block:: console

    pip uninstall mocafe

Notice that *mocafe* has some dependencies, such as ``tqdm``, ``pandas``, and so on, that won't be automatically
removed with the command above. To remove them, you need to tell ``pip`` to do so.

This is what you need to type to remove *moacfe* with all its dependencies:

.. code-block:: console

    pip uninstall mocafe numpy pandas pandas-ods-reader tqdm

Of course, if you use any of the packages listed above for other purposes you should not remove them.

.. _remove-fenics-apt:

Remove FEniCS with apt
^^^^^^^^^^^^^^^^^^^^^^
In case you installed FEniCS using apt, you can just remove it with all its dependencies using ``apt autoremove``:

.. code-block:: console

    sudo apt autoremove fenics -y

Notes
-----
