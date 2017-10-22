Written for Python 3 using gcc 5.4. A C++14 compiler required.

Can be install with pip. Requires ALE (and it's accompanying dependencies) to work as well as numpy. Both will be installed as dependencies via pip:

```
pip install --install-option="--prefix=$PREFIX_PATH" git+https://github.com/neuro-evolution/alectrnn.git
```

`$PREFIX_PATH$` will specify the install location of the package (optional). Installing directly from the repo will require git to be installed as well (i.e. for the git+ command).

ALE supports a visual interface via SDL. If you want to install using the visual interface then you need to first install the libsdl1.2-dev library. By default SDL is not installed with ALE. It can be enabled via the `--with-sdl` option:

```
pip install --install-option="--with-sdl" git+https://github.com/neuro-evolution/alectrnn.git
```
