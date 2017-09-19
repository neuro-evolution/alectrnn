Written for Python 3 using gcc 5.4. A C++14 compiler required.

Can be install with pip. Requires ALE (and it's accompanying dependencies) to work as well as numpy. Both will be installed as dependencies via pip:

```
pip install --install-option="--prefix=$PREFIX_PATH" git+https://github.com/neuro-evolution/alectrnn.git
```

`$PREFIX_PATH$` will specify the install location of the package (optional). Installing directly from the repo will require git to be installed as well (i.e. for the git+ command).