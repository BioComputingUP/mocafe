# _mocafe_: modeling cancer with FEniCS

> A Python package based on [FEniCS](https://fenicsproject.org/) to develop and simulate 
> Phase Field mathematical models for cancer

The purpose of _mocafe_ is to provide an open source and efficient implementation of legacy Phase Field mathematical 
models of cancer development and angiogenesis. Use _mocafe_ to:

* reproduce legacy cancer phase field model (also in parallel with `MPI`);
* understand them better inspecting the code, and changing the parameters;
* integrate these models in yours [FEniCS](https://fenicsproject.org/) scripts.

![](header.png)

## Installation

### Linux quick setup (_not_ recommended)

To have access to parallel computation, first install MPI:

```sh
sudo apt install openmpi-bin
```

And test it using `mpirun --version`. If everything worked out well, you should get something like:

```sh
mpirun (Open MPI) 4.0.3

Report bugs to http://www.open-mpi.org/community/help/
```

Then, install FEniCS using `apt-get`:

```sh
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

Test if FEniCS has been correctly installed trying to import it on IPython:

```sh
python3
Type "help", "copyright", "credits" or "license" for more information.
>>> import fenics
>>>
```

Install _mocafe_ using `pip`:

```sh
pip3 install git+https://github.com/fpradelli94/mocafe#egg=mocafe
```

Test it executing the main script:

```sh
python3 -m mocafe
```

I everything is properly working, the output should be:

```
Your mocafe is ready!
```

## Usage example

See demo gallery in [docs](https://fpradelli94.github.io/mocafe/)

## Development setup

Todo

## Release History

Todo

## Meta

Franco Pradelli – email

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/fpradelli94/mocafe](https://github.com/fpradelli94/mocafe)

## Contributing

Todo

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
