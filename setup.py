from setuptools import setup

setup(
  name='exoscene',
  description='Library for simulating direct images of exoplanetary systems.',
  version='1.0',
  author='Neil T. Zimmerman',
  author_email='neil.t.zimmerman@nasa.gov',

  packages=['exoscene'],
  package_data={'exoscene': ['./data/bpgs/bpgs_*.fits',
                             './data/bpgs/bpgs_readme.csv',
                             './data/cgi_hlc_psf/*.fits']},

  install_requires=[
      "astropy >= 4.0",
      "numpy",
      "scipy",
      "scikit-image",
      "pandas"
  ],
)

