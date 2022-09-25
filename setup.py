from setuptools import setup

setup(
    name='metannvis',
    version='0.07',
    packages=['src', 'src.metannvis', 'src.metannvis.methods', 'src.metannvis.toolsets', 'src.metannvis.frameworks',
              'src.metannvis.translations', 'src.unittests'],
    url='https://github.com/sfluegel05/metaNNvis',
    license='MIT',
    author='Simon Flügel',
    author_email='sfluegel@ovgu.de',
    description=' MetaNNvis is a tool for accessing introspection methods for neural networks regardless of the framework in which the neural network has been built.'
)
