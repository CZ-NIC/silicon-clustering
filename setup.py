from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='silicon-clustering',
      version='0.3.1',
      description='Single-linkage connectivity clustering by cosine similarity',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='connectivity clustering similarity PCA',
      url='https://github.com/CZ-NIC/silicon-clustering',
      author='Tomas Gavenciak',
      author_email='gavento@ucw.cz',
      license='MIT',
      packages=['silicon'],
      install_requires=[
          'scipy>=0.19',
          'numpy>=1.10',
          'scikit-learn>=0.18',
          'six>=1.10',
      ],
      test_suite='nose.collector',
      tests_require=['nose>=1.3', 'matplotlib'],
      include_package_data=True,
      zip_safe=True)
