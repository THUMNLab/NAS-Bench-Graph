from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

''' https://packaging.python.org/guides/distributing-packages-using-setuptools/ '''
''' https://setuptools.readthedocs.io/en/latest/ '''
setup(
    name='nas-bench-graph',
    version='1.0',
    author='THUMNLab/aglteam',
    maintainer='THUMNLab/aglteam',
    author_email='autogl@tsinghua.edu.cn',
    description='NAS benchmark for graph data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(),
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='~=3.6',
    # https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6"
    ],
    # https://setuptools.readthedocs.io/en/latest/userguide/dependency_management.html
    # note that setup_requires and tests_require are deprecated
    install_requires=[
        'pickle',
        'torch',
        'modulefinder',
        'nni',
        'sys'
    ]
)
