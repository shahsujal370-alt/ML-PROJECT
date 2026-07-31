from typing import List

from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Return the list of requirements from a requirements file."""
    requirements = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return [req for req in requirements if req and not req.startswith("#")]


setup(
    name="ml-project",
    version="0.0.1",
    description="End-to-end machine learning project package",
    author="Sujal",
    author_email="shahsujal370@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    include_package_data=True,
)