import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setuptools.setup(
    name="filterpicker",
    version="1.0.1",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@erdw.ethz.com",
    description="Python implementation of the A.Lomax Filter Picker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ethz.ch/mbagagli/filterpicker",
    python_requires='>=3.6',
    install_requires=required_list,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Intended Audience :: Science/Research",
    ],
)
