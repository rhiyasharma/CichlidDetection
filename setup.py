from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='CichlidDetection',
      version='0.1',
      description='',
      long_description=readme(),
      url='https://github.com/ptmcgrat/CichlidDetection',
      license='MIT',
      packages=['CichlidDetection'],
      install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'shapely', 'pillow', 'opencv', 'pytorch',
                        'torchvision'],
      include_package_data=True,
      zip_safe=False
      )
