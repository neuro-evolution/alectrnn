from setuptools import setup, Extension, Command
import setuptools.command.develop
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import subprocess
import sys
import os


def run_ale_install_script(with_sdl, lib_path):
    """
    Downloads and installs ALE to installation directory
    """
    if with_sdl and lib_path:
        build_libs_cmd = ['bash', 'alectrnn/ale_install.sh', '--with-sdl',
                          '--lib-path', lib_path]
    elif with_sdl:
        build_libs_cmd = ['bash', 'alectrnn/ale_install.sh', '--with-sdl']

    elif lib_path:
        build_libs_cmd = ['bash', 'alectrnn/ale_install.sh',
                          '--lib-path', lib_path]
    else:
        build_libs_cmd = ['bash', 'alectrnn/ale_install.sh']

    if subprocess.call(build_libs_cmd) != 0:
        sys.exit("Failed to build ALE dependencies")


class install(setuptools.command.install.install):
    user_options = setuptools.command.install.install.user_options \
                    + [("with-sdl", None, "Enable SDL in ALE"),
                       ("lib-path=", None, "Install ALE from path")]

    def initialize_options(self):
        setuptools.command.install.install.initialize_options(self)
        self.with_sdl = None
        self.lib_path = None

    def finalize_options(self):
        setuptools.command.install.install.finalize_options(self)

    def run(self):
        run_ale_install_script(self.with_sdl, self.lib_path)
        setuptools.command.install.install.run(self)


class develop(setuptools.command.develop.develop):
    user_options = setuptools.command.develop.develop.user_options \
                    + [("with-sdl", None, "Enable SDL in ALE"),
                       ("lib-path=", None, "Install ALE from path")]

    def initialize_options(self):
        setuptools.command.develop.develop.initialize_options(self)
        self.with_sdl = None
        self.lib_path = None

    def finalize_options(self):
        setuptools.command.develop.develop.finalize_options(self)

    def run(self):
        run_ale_install_script(self.with_sdl, self.lib_path)
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        try:
            import numpy as np
            numpy_include_path = np.get_include()# + '/numpy'
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
extra_compile_args = ['-std=c++14', '-Wno-write-strings', '-Wno-undef']

# Includes
include_dirs = []
cwd = os.path.dirname(os.path.abspath(__file__))
ale_install_path = cwd + "/alectrnn/alelib"
include_dirs += [
    cwd,
    ale_install_path + '/include/ale',
    cwd + '/alectrnn/agents',
    cwd + '/alectrnn/common',
    cwd + '/alectrnn/nervous_system',
    cwd + '/alectrnn/random',
    cwd + '/alectrnn/controllers',
    cwd + '/alectrnn/objectives',
    cwd + '/alectrnn'  # For Eigen
]

# Libraries
lib_path = os.path.join(cwd, "alectrnn", "alelib", "lib")
library_dirs = [lib_path]
ALE_LIB = os.path.join(lib_path, "libale.so")
main_link_args = [ALE_LIB,"-lstdc++"]
main_libraries = ['ale']
extra_link_args = ['-Wl,--verbose']

# Sources
ale_sources = [
    "alectrnn/common/ale_generator.cpp"
]

agent_sources = [
    "alectrnn/agents/agent_generator.cpp",
    "alectrnn/agents/player_agent.cpp",
    "alectrnn/agents/ctrnn_agent.cpp",
    "alectrnn/agents/nervous_system_agent.cpp",
    "alectrnn/common/network_constructor.cpp",
    "alectrnn/common/ctrnn.cpp",
    "alectrnn/common/screen_preprocessing.cpp",
    "alectrnn/agents/soft_max_agent.cpp",
    "alectrnn/agents/shared_motor_agent.cpp",
    "alectrnn/agents/reward_mod_agent.cpp"
]

objective_sources = [
    "alectrnn/objectives/objective.cpp",
    "alectrnn/common/capi_tools.cpp",
    "alectrnn/agents/player_agent.cpp",
    "alectrnn/agents/nervous_system_agent.cpp",
    "alectrnn/common/screen_preprocessing.cpp",
    "alectrnn/controllers/controller.cpp"
]

layer_sources = [
    "alectrnn/nervous_system/layer_generator.cpp",
    "alectrnn/common/capi_tools.cpp"
]

nn_sources = [
    "alectrnn/nervous_system/nervous_system_generator.cpp",
    "alectrnn/common/capi_tools.cpp"
]

nn_handler_sources = [
    "alectrnn/nervous_system/nervous_system_handler.cpp",
    "alectrnn/common/capi_tools.cpp"
]

ale_handler_sources = [
    "alectrnn/common/ale_handler.cpp"
]

agent_handler_sources = [
    "alectrnn/agents/agent_handler.cpp",
    "alectrnn/agents/player_agent.cpp",
    "alectrnn/agents/nervous_system_agent.cpp",
    "alectrnn/common/screen_preprocessing.cpp"
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
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

agent_module = Extension('agent_generator',
                    language = "c++14",
                    sources=agent_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

objective_module = Extension('objective',
                    language = "c++14",
                    sources=objective_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

layer_module = Extension('layer_generator',
                    language = "c++14",
                    sources=layer_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

nn_module = Extension('nn_generator',
                    language = "c++14",
                    sources=nn_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

nn_handler_module = Extension('nn_handler',
                    language = "c++14",
                    sources=nn_handler_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

ale_handler = Extension('ale_handler',
                    language = "c++14",
                    sources=ale_handler_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

agent_handler = Extension('agent_handler',
                    language = "c++14",
                    sources=agent_handler_sources,
                    libraries=main_libraries,
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_link_args=extra_link_args + main_link_args
                        + ['-Wl,-rpath,$ORIGIN/alelib/lib'])

setup(name=PACKAGE_NAME,
      version='1.1',
      author='Nathaniel Rodriguez',
      cmdclass={'build_ext': build_ext,
                'install': install,
                'develop': develop
                },
      description='A wrapper for a ctrnn implementation of ALE',
      url='https://github.com/neuro-evolution/alectrnn.git',
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[ale_module, agent_module, objective_module,
                   layer_module, nn_module, nn_handler_module,
                   ale_handler, agent_handler],
      package_data={PACKAGE_NAME: [
        'roms/*.bin',
        'alelib/bin/ale',
        'alelib/lib/*.so',
        'alelib/include/ale/*.h*',
        'alelib/include/ale/common/*.h*',
        'alelib/include/ale/controllers/*.h*',
        'alelib/include/ale/emucore/*.h*',
        'alelib/include/ale/emucore/m6502/src/*.h*',
        'alelib/include/ale/emucore/m6502/src/bspf/src/*.h*',
        'alelib/include/ale/environment/*.h*',
        'alelib/include/ale/external/TinyMT/*.h*',
        'alelib/include/ale/games/*.h*',
        'alelib/include/ale/games/supported/*.h*',
        'alelib/include/ale/os_dependent/*.h*']})
