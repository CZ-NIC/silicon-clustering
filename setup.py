from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='cosinesimilarity',
      version='0.1',
      description='Clustering by cosine similarity accelerated by PCA',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
      ],
      keywords='clustering cosine similarity PCA',
      url='',
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

