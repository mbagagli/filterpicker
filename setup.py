from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setup(
    name="filterpicker",
    version="1.0.4",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@erdw.ethz.ch",
    description="Python implementation of the A.Lomax Filter Picker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ethz.ch/mbagagli/filterpicker",
    python_requires='>=3.5',
    install_requires=required_list,
    setup_requires=['wheel'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        'console_scripts': [
            'obspy_script=filterpicker.cli.obspy_script:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
