from setuptools import setup, find_packages

setup(
    name='ATI_FTsensor',
    version='0.1.0',
    packages=['ATI_FTsensor'],
    package_dir={'ATI_FTsensor': '.'},
    package_data={'ATI_FTsensor': ['FT44764/*.cal']},
    include_package_data=True,
    description='ATI Force/Torque Sensor Python Interface',
    author='besically it P\'Joey not me',
    author_email='-',
    url='https://github.com/Robotics09Lab/ATI_FT-sensor',
    install_requires=[
        'numpy',
        'nidaqmx',
        'atiiaftt',
    ],
)
