from setuptools import setup
#import anemoi-analyse

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='anemoi-analyse',
      version="0.0.1",
      description='Anemoi output analysing tools',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/evenmn/anemoi-analyse',
      author='Even Marius Nordhagen',
      author_email='even.nordhagen@gmail.com',
      license='GPL-v3',
      packages=['anemoi_analyse'],
      include_package_data=True,
      install_requires=['matplotlib', 'scipy'],
      entry_points={
          'console_scripts': [
              'loss-plotter=anemoi_analyse.cmd_interface:run',
          ]
      },
      zip_safe=False,
)
