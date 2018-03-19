Written for Python 3 using g++ 5.4. A C++14 compiler required.

Requirements: Numpy, ALE, CMake (for ALE)

Can be installed with pip. ALE and Numpy dependencies will be handled by pip. Can be installed using:

```
pip install --install-option="--prefix=$PREFIX_PATH" git+https://github.com/neuro-evolution/alectrnn.git
```

`$PREFIX_PATH$` will specify the install location of the package (optional). Installing directly from the repo will require git to be installed as well (i.e. for the git+ command).

ALE supports a visual interface via SDL. If you want to install using the visual interface then you need to first install the libsdl1.2-dev library. By default SDL is not installed with ALE. It can be enabled via the `--with-sdl` option:

```
pip install --install-option="--with-sdl" git+https://github.com/neuro-evolution/alectrnn.git
```

This library is based on float (single precision values). If using with numpy, make sure to use dtype=float32 instead of the default float64

Note: the --install-option="--lib-path=/path/to/ale-master" can allow compilation using a local download of the ALE source code.
Note: the ale_install.sh file directly uses gcc and g++ environment variables to define CC and CXX for cmake.