from setuptools import setup

with open("README.md", "r") as fh:
        long_description = fh.read()

setup(
  name='exoscene',
  version='1.2',
  description='Library for simulating direct images of exoplanetary systems.',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/nasa/exoscene",
  author='Neil T. Zimmerman',
  author_email='neil.t.zimmerman@nasa.gov',

  packages=['exoscene'],
  package_data={'exoscene': ['./data/bpgs/bpgs_*.fits',
                             './data/bpgs/bpgs_readme.csv',
                             './data/cgi_hlc_psf/*.fits']},

  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved",
  ],

  python_requires='>=3.6',
  install_requires=[
      "astropy >= 4.0",
      "numpy",
      "scipy",
      "scikit-image",
      "pandas",
      "ipython",
      "matplotlib",
      "notebook"
  ],
)

