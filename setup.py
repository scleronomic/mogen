from setuptools import setup, find_packages

with open("docs/README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mogen",
    version="0.0.1",
    author="Johannes Tenhumberg",
    author_email="johannes.tenhumberg@gmail.com",
    description="MoGen - Motion Generation - Optimization-based Motion Planning in Python - Data Generation",
    long_description=long_description,
    url="https://github.com/scleronomic/mogen",
    packages=find_packages(),
    install_requires=[
        'wzk @ git+https://github.com/scleronomic/wzk',
        'rokin @ git+https://github.com/scleronomic/rokin',
        'mopla @ git+https://github.com/scleronomic/mopla',
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
