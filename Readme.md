# densityNN

This project is going to be used for studying density-based methods used in AI, including Gaussian Processes, variational flows, etc. We shall explore different ways of exploring the errors that neural networks might cause in a meaningful manner. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. his project, like all my other projects uses a particular project structure. Details of how to use this project is present in the [Wiki page](https://github.com/sankhaMukherjee/densityNN/wiki). Please visit that to understand the project sis structured.

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for \*nix-like systems. This is currently tested in the following system: `Ubuntu 18.10`. 

For installation, first close this repository, and generate the virtual environment required for running the programs. 

This project framework uses [venv](https://docs.python.org/3/library/venv.html) for maintaining virtual environments. Please familiarize yourself with [venv](https://docs.python.org/3/library/venv.html) before working with this repository. You do not want to contaminate your system python while working with this repository.

A convenient script for doing this is present in the file [`bin/vEnv.sh`](../master/bin/vEnv.sh). This is automatically do the following things:

1. Generate a virtual environment
2. activate this environment
3. install all required libraries
4. deactivate the virtual environment and return to the prompt. 

At this point you are ready to run programs. However, remember that you will need to activate the virtual environment any time you use the program.

For activating your virtual environment, type `source env/bin/activate` in the project folder in [bash](https://www.gnu.org/software/bash/) or `source env/bin/activate.fish` if you are using the [fish](https://fishshell.com/) shell.
For deactivating, just type deactivate in your shell

## Documentation

Detailed documentation is present in the [Wiki](https://github.com/sankhaMukherjee/densityNN/wiki). The API generated is present in the repos website [website](https://sankhamukherjee.github.io/densityNN/index.html). 

Since this repo mainly comprises of a set of tutorials, the main page of the tutorials may be found [here](https://github.com/sankhaMukherjee/densityNN/wiki/done). This comprises of a set of tutorials that can be used for exploring density-based AI methodologies.

Here are some quick links:

| API Reference  |
|----------------|
| [API Main page](https://sankhamukherjee.github.io/densityNN/index.html) | 
| [Modules](https://sankhamukherjee.github.io/densityNN/modules.html) |
| [lib](https://sankhamukherjee.github.io/densityNN/lib.html) |
| Indices: [Index](https://sankhamukherjee.github.io/densityNN/genindex.html) \| [Module Index](https://sankhamukherjee.github.io/densityNN/py-modindex.html)  |




## Contributing

Plaease see detailed instructions of how to contribute to this project [here](https://github.com/sankhaMukherjee/densityNN/wiki/Contributing).

## Authors

Sankha S. Mukherjee - Initial work (2019)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

 
## References:

1. Murphy K. P., Machine Learning: a Probabilistic Perspective,  ISBN-13: 978-0262018029, ISBN-10: 0262018020 
2. Nando de Freitas, [Machine Learning 2013, Video Lectures](https://www.youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6)