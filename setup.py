from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='silicon-clustering',
      version='0.1',
      description='Single-linkage connectivity clustering by cosine similarity',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='connectivity clustering similarity PCA',
      url='https://github.com/CZ-NIC/silicon-clustering',
      author='Tomas Gavenciak',
      author_email='tomas.gavenciak@nic.cz',
      license='MIT',
      packages=['cosinesimilarity'],
      install_requires=[
          'scipy',
          'numpy',
          'sklearn',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=True)
