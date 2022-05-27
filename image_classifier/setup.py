from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.2'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='silknow_image_classification',
    version=__version__,
    description='A python package that can perform image classification.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Dorozynski/silknow_image_classification',
    download_url='https://github.com/Dorozynski/silknow_image_classification/tarball/' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='LUH',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='dorozynski@ipi.uni-hannover.de'
)
