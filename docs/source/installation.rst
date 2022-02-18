Installation
============
Currently, *mocafe* has been tested only on Linux and on Windows, using the Windows Subsystem for Linux (WSL). In both
cases the installation instructions are identical.

*Notice:* WSL is available for Windows 10 onwards.

Installing *mocafe* in Singularity container (**recommended**)
--------------------------------------------------------------
A FEniCS Docker container is available at `quay.io/fenicsproject/stable <quay.io/fenicsproject/stable>`_; however,
Docker does not provide a full support to MPI. Thus, the best way to use FEniCS and *mocafe* in a container is to use
`Singularity <https://sylabs.io/docs/>`_, which supports also Docker containers.

If you don't have Singularity installed, just follow the instruction provided at the official `documentation page
for SingularityCE <https://github.com/sylabs/singularity>`_.

If you have Singularity, just build a FEniCS container typing:

.. code-block:: console

    singularity build fenics.sif docker:quay.io/fenicsproject/stable:latest

It may take some time to do all the necessary operations.

Now, the plan is to use our FEniCS singularity container (``fenics.sif``) and install *mocafe* inside it using ``pip``.
However, using ``pip`` inside a singularity container is not enough to separate your host system python packages from
those you're using in your container. A good solution to this problem is provided `here
<https://git.its.aau.dk/CLAAUDIA/docs_aicloud/src/branch/master/aicloud_slurm/pip_in_containers/pip_in_containers.md>`__
and we're going to use this approach in the following.

First of all, create a local directory for the *mocafe* python package. If your Linux username is ``username``, you can
simpy do something like this:

.. code-block:: console

    mkdir /home/username/.mocafe-local

Then enter the Singularity container using the ``shell`` command, using the ``--bind`` option bind the directory we
just created with the directory ``/home/username/.local``. The reason for that will become clear later.

.. code-block:: console

    singularity shell --bind /home/username/.mocafe-local:/home/username/.local fenics.sif

Now you are inside a container with FEniCS and MPI installed. So, you just require to install mocafe
using pip:

.. code-block:: console

    pip3 install --user git+https://github.com/fpradelli94/mocafe#egg=mocafe

*Note*: some warnings regarding the pip version inside the container may occur. You can ignore them.

Now *mocafe* with all its dependencies is installed on your container! Test it typing:

.. code-block:: console

    python3 -m mocafe

If everything is working properly, you should see the output message:

.. code-block:: console

    Your mocafe is ready!

If everything worked out well, you might wander why we recommended you to create the folder
``/home/username/.mocafe-local``. The fact is that the default behaviour of ``pip install --user`` is to
place the python package you're installing in the folder ``/home/username/.local``. However, being inside
a singularity container doesn't stop ``pip`` to use your host system ``.local`` directory, with the result that
you have a python package installed inside a container on your host system as well. Thus, we just created
a new directory (``.mocafe-local``) and we told singularity to just consider that directory as the ``.local``
directory for the container, using the option ``--bind`` (full documentation `here
<https://sylabs.io/guides/latest/user-guide/bind_paths_and_mounts.html>`__). So now we have all our python
packages secured in their folder and we don't risk to mix up different things.


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


Uninstalling
------------
In case you want to remove *mocafe* and its dependencies from your system, you just need to follow the instructions
provided below. Notice that uninstalling instruction change depending on the installation procedure you followed.

In case you installed *mocafe* using singularity:

1. Uninstall *mocafe* and its python dependencies using ``pip uninstall`` (inside the container); see section :ref:`uninstalling-mocafe`
2. In case you created it, remove the folder ``/home/username/.mocafe-local`` using ``rm -r /home/username/.mocafe-local``
3. Remove the FEniCS container (in case you don't need it anymore); see section :ref:`remove-fenics-container`
4. Remove Singularity (in case you don't need it anymore); see section :ref:`remove-singularity`

In case you installed *mocafe* using ``apt``:

1. Uninstall *mocafe* and its python dependencies using ``pip uninstall``; see section :ref:`uninstalling-mocafe`
2. Remove FEniCS using ``apt autoremove``; see section :ref:`remove-fenics-apt`

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

.. _remove-fenics-container:

Remove FEniCS container
^^^^^^^^^^^^^^^^^^^^^^^
To remove correctly the FEniCS container from your system, you first need to remove all the cached data:

.. code-block:: console

    singularity cache clean fenics.sif

Then, you can simply remove the ``fenics.sif`` file:

.. code-block:: console

    rm fenics.sif

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

.. _remove-fenics-apt:

Remove FEniCS with apt
^^^^^^^^^^^^^^^^^^^^^^
In case you installed FEniCS using apt, you can just remove it with all its dependencies using ``apt autoremove``:

.. code-block:: console

    sudo apt autoremove fenics -y