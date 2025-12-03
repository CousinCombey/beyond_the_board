from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="beyond_the_board",
    version="0.1.0",
    author="Beyond The Board Team",
    description="Chess position evaluation using machine learning",
    long_description_content_type="text/markdown",
    # url="https://github.com/CousinCombey/beyond_the_board",
    packages=find_packages(),
    install_requires=requirements,
    test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False)
