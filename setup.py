from setuptools import setup

setup(
  name='exoscene',
  version='1.0',
  author='Neil T. Zimmerman',
  author_email='neil.t.zimmerman@nasa.gov',
  packages=['exoscene'],
  description='Library for simulating direct images of exoplanetary systems.',
  install_requires=[
      "astropy >= 4.0",
      "numpy",
      "scipy",
      "skimage",
      "pandas"
  ],
)

