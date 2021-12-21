import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="mocafe",
    version="0.0.1",
    description="A Python package based on FEniCS to model and simulate Phase Field models for cancer",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/fpradelli94/mocafe",
    author="Franco Pradelli",
    license="MIT",
    packages=find_packages(exclude=("test", "demo", "docs", "projects", ".cache")),
    include_package_data=True,
    install_requires=["numpy", "pandas", "pandas-ods-reader", "tqdm"]
)
