from setuptools import setup, Extension, Command
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import subprocess
import sys
import os

def run_ale_install_script():
    # If we need to give compiler options this is where we could pass to script
    build_libs_cmd = ['bash', 'ale_install.sh']
    if subprocess.call(build_libs_cmd) != 0:
        sys.exit("Failed to build ALE dependencies")

class build_ale(Command):
    def run(self):
        run_ale_install_script()

class install(setuptools.command.install.install):
    def run(self):
        self.run_command('build_ale')
        setuptools.command.install.install.run(self)

class build(distutils.command.build.build):
    sub_commands = [
        ('build_ale', lambda self: True),
        ] + distutils.command.build.build.sub_commands

class build_ext(setuptools.command.build_ext.build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        try:
            import numpy as np
            numpy_include_path = np.get_include() + '/numpy'
            if numpy_include_path not in self.include_dirs:
                self.include_dirs.append(numpy_include_path)
        except ImportError:
            sys.exit("Error: Numpy required")

        # Call original build_ext command
        setuptools.command.build_ext.build_ext.run(self)

# Remove setuptools dumb c warnings
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Compiler settings
extra_compile_args = ['-std=c++14']

# Includes
include_dirs = []
cwd = os.path.dirname(os.path.abspath(__file__))
ale_install_path = cwd + "/alelib"
include_dirs += [
    cwd,
    ale_install_path + '/include/ale',
    # ale_install_path + '/include/ale/common',
    # ale_install_path + '/include/ale/controllers',
    # ale_install_path + '/include/ale/emucore',
    # ale_install_path + '/include/ale/emucore/m6502',
    # ale_install_path + '/include/ale/emucore/m6502/src',
    # ale_install_path + '/include/ale/emucore/m6502/src/bspf/src',
    # ale_install_path + '/include/ale/environment',
    # ale_install_path + '/include/ale/external/TinyMT',
    # ale_install_path + '/include/ale/games',
    # ale_install_path + '/include/ale/games/supported',
    # ale_install_path + '/include/ale/os_dependent',
    cwd + '/agents',
    cwd + '/common',
    cwd + '/controllers',
    cwd + '/objectives'
]

# Libraries
lib_path = os.path.join(cwd, "alelib", "lib")
library_dirs = []
library_dirs.append(lib_path)
ALE_LIB = os.path.join(lib_path, "libale.so")
ALEC_LIB = os.path.join(lib_path, "libale_c.so")
main_link_args = [ALE_LIB, ALEC_LIB]
main_libraries = ['libale', 'libale_c']
extra_link_args = []

# Sources
ale_sources = [
    "common/ale_generator.cpp"    
]
agent_sources = [
    "agents/agent_generator.cpp",
    "agents/player_agent.cpp",
    "agents/ctrnn_agent.cpp",
    "common/network_generator.cpp",
    "common/nervous_system.cpp",
    "common/screen_preprocessing.cpp"
]
objective_sources = [
    "objectives/total_cost_objective.cpp",
    "common/utilities.cpp",
    "agents/player_agent.cpp",
    "controllers/controller.cpp"
]

PACKAGE_NAME = 'alectrnn'

ale_module = Extension('ale_generator',
                    language="c++14",
                    sources=ale_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/../lib'])

agent_module = Extension('agent_generator',
                    language = "c++14",
                    sources=agent_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/../lib'])

objective_module = Extension('total_cost_objective',
                    language = "c++14",
                    sources=objective_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/../lib'])

setup(name=PACKAGE_NAME,
      version='1.0',
      author='Nathaniel Rodriguez',
      cmdclass = {'build_ext': build_ext, 'install': install,
                  'build_ale': build_ale, 'build_py':build_py},
      description='A wrapper for a ctrnn implementation of ALE',
      url='https://github.com/neuro-evolution/alectrnn.git',
      install_requires=[
          'numpy'
      ],
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[ale_module, agent_module, objective_module],
      include_package_data=True)