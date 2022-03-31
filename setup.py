from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "sklearn",
    "pickle"
    ]

tests_require = [
    "pytest",
    "os",
    "hypothesis"
    ]

setup(
      name = 'neuralnet',
      version = '1.0',
      description = 'Implementation of an artificial neural network from scratch',
      author = 'Filippo Antonio Schiazza',
      author_email = 'filipposchiazza97@gmail.com',
      url = 'https://github.com/filipposchiazza/Neural-Network-Project',
      license = 'None',
      packages = find_packages(),
      install_requires = install_requires,
      tests_require = tests_require,
      
      
)

