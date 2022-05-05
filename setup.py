from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='neuralnet',
   version='1.0',
   description='Implementation of a neural network, using only numpy library',
   license="MIT",
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='Filippo Schiazza',
   author_email='filipposchiazza97@gmail.com',
   url="https://github.com/filipposchiazza/NeuralNetworkFromScratch",
   packages=['neuralnet'],  #same as name
   install_requires=['numpy', 'sklearn', 'hypothesis'], #external packages as dependencies
   scripts=[
            'scripts/binary_classification.py',
            'scripts/classification_Digits.py',
            'scripts/classification_Iris.py'
           ]
)
