This is a Python 3 module designed to allow for a high level interface with neural network agents that play Atari games (via ALE). The Python API is divided into 3 main sections: the `ALEExperiment` class which allows high level setup of parameters for a complete run of ALE with some neural network and objective function (training is not a part of this module). `handlers.py` deals with the interface between the c++ code and python, and `analysis_tools.py` includes various plotting functions for displaying neuron state, parameter distribution, and other tools.

The c++ portion of the code follows the following structure: `controller` and `player_agent` classes define the interface of the agent with ALE. `objective.cpp` defines available objective functions. The neural networks are constructed via the `NervousSystem` which contains `Layers`, which in turn contains `Integrators` and `Activators` of various types, which define the structure of the network, its parameters, and neuronal dynamics. The Python `NervousSystem` handler defines the API for the lower level c++ code and allows construction of various types of neural networks. The user can create convolutional layers, recurrent layers, reservoir layers, all-to-all layers, and hybrid layers of the various types. For details about layer types, see the doc strings for the `NervousSystem` class in `handlers.py` or use `help` in the Python terminal.  

Written for Python 3 using g++ 5.4. A C++14 compiler required.

Requirements: Numpy, ALE, CMake (for ALE)

Can be installed with pip. ALE (a modified forked version) and Numpy dependencies will be handled by pip. Can be installed using:

```
pip install --install-option="--prefix=$PREFIX_PATH" git+https://github.com/nathaniel-rodriguez/alectrnn.git
```

`$PREFIX_PATH$` will specify the install location of the package (optional). Installing directly from the repo will require git to be installed as well (i.e. for the git+ command).

ALE supports a visual interface via SDL. If you want to install using the visual interface then you need to first install the libsdl1.2-dev library. By default SDL is not installed with ALE. It can be enabled via the `--with-sdl` option:

```
pip install --install-option="--with-sdl" git+https://github.com/nathaniel-rodriguez/alectrnn.git
```

This library is based on float (single precision values). If using with numpy, make sure to use dtype=float32 instead of the default float64.

Additional notes:

- The --install-option="--lib-path=/path/to/ale-master" can allow compilation using a local download of the [ALE source code](https://github.com/Nathaniel-Rodriguez/arcade-learning-environment).

- The ale_install.sh file directly uses gcc and g++ environment variables to define CC and CXX for cmake.
 
- To make animations from the `analysis_tools` module, ffmpeg needs to be installed for matplotlib to use.

- The `alectrnn_experiment_template.py` requires `evostrat` to run. It can be installed from here: https://github.com/Nathaniel-Rodriguez/evostrat.git. See the template and python `help` command for examples and documentation.