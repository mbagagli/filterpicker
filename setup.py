from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setup(
    name="filterpicker",
    version="1.1.0",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@ingv.it",
    description="Python implementation of the A.Lomax Filter-Picker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbagagli/filterpicker",
    python_requires='>=3.6',
    install_requires=required_list,
    setup_requires=['wheel'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    zip_safe=False,
    scripts=['bin/run_filter_picker.py']
)
