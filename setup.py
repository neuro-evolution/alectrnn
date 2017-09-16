from setuptools import setup, Extension

def readme():
    with open('README.rst') as f:
        return f.read()

PACKAGE_NAME = 'alectrnn'

objective_module = Extension('objectives',
                    language = "c++",
                    sources = ['objectives/total_cost_objective.cpp'])
agent_module = Extension('agents',
                    language = "c++",
                    sources = ['agents/agent_generator.cpp'])
ale_module = Extension('ale',
                    language = "c++",
                    sources = ['common/ale_generator.cpp'])

setup(name=PACKAGE_NAME,
      version='1.0',
      author='Nathaniel Rodriguez',
      description='A wrapper for a ctrnn implementation of ALE',
      url='https://github.com/neuro-evolution/alectrnn.git',
      install_requires=[
          'numpy'
      ],
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[objective_module, agent_module, ale_module],
      include_package_data=True)