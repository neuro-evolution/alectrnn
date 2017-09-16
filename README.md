Written for Python 3 using gcc 5.4. A C++14 compiler required.

Can be install with pip. Requires ALE (and it's accompanying dependencies) to work. See [this github repo](https://github.com/neuro-evolution/arcade-learning-environment) for the ALE fork that works with ALECTRNN. Once you have installed ALE (see the directions on ALE github page) you will need to add the include path as an argument to pip, so pip can link the shared library with ALECTRNN. This can be done with the following line:

```
pip install --global-option=build_ext --global-option="-I/$INCLUDE_PATH$" --global-option="-L/$LIBRARY_PATH$" --install-option="--prefix=$PREFIX_PATH" git+https://github.com/neuro-evolution/alectrnn.git
```

`$INCLUDE_PATH$` and `$LIBRARY_PATH$` should be the paths to ALEs include directory and shared library file respectively. `$PREFIX_PATH$` will specify the install location of the package. Installing directly from the repo will require git to be installed as well (i.e. for the git+ command).