from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pgs_dsnd_distributions',
      version='0.6',
      description='Gaussian & Binomial distributions',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/paul-stubley/Udacity/tree/master/Project_2__Creating_a_PyPi_package",
      packages=['pgs_dsnd_distributions'],
      zip_safe=False)
