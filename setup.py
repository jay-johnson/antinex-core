import os
import sys
import warnings
import unittest

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


cur_path, cur_script = os.path.split(sys.argv[0])
os.chdir(os.path.abspath(cur_path))

requires_that_fail_on_rtd = [
    "h5py",
    "keras",
    "tables",
    "tensorflow"
]

install_requires = [
    "antinex-utils",
    "celery>=4.1.0",
    "celery-connectors",
    "celery-loaders",
    "colorlog",
    "coverage",
    "docker-compose",
    "flake8<=3.4.1",
    "future",
    "matplotlib",
    "numpy>=1.13.3",
    "pandas",
    "pep8>=1.7.1",
    "pipenv",
    "pydocstyle<=2.3.1",
    "pydocstyle",
    "pylint",
    "recommonmark",
    "requests",
    "scikit-learn",
    "seaborn",
    "sphinx",
    "sphinx-autobuild",
    "sphinx_bootstrap_theme",
    "sphinx_rtd_theme",
    "tox",
    "tqdm",
    "unittest2",
    "mock"
]

# if not on readthedocs.io get all the pips:
if os.getenv("READTHEDOCS", "") == "":
    install_requires = install_requires + requires_that_fail_on_rtd

if sys.version_info < (3, 5):
    warnings.warn(
        "Less than Python 3.5 is not supported.",
        DeprecationWarning)


# Do not import antinex_core module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "antinex_core"))

setup(
    name="antinex-core",
    cmdclass={"test": PyTest},
    version="1.0.40",
    description=("AntiNex publisher-subscriber core for processing "
                 "training and prediction requests for deep neural "
                 "networks to detect network exploits using Keras "
                 "and Tensorflow in near real-time."),
    long_description=("AntiNex publisher-subscriber core for processing "
                      "training and prediction requests for deep neural "
                      "networks to detect network exploits using Keras "
                      "and Tensorflow in near real-time."),
    author="Jay Johnson",
    author_email="jay.p.h.johnson@gmail.com",
    url="https://github.com/jay-johnson/antinex-core",
    packages=[
        "antinex_core",
        "antinex_core.scripts",
        "antinex_core.log"
    ],
    package_data={},
    install_requires=install_requires,
    test_suite="setup.antinex_core_test_suite",
    tests_require=[
        "pytest"
    ],
    scripts=[
        "./run-antinex-core.sh",
        "./antinex_core/scripts/publish_train_request.py",
        "./antinex_core/scripts/publish_predict_request.py",
        "./antinex_core/scripts/antinex_scaler_django.py",
        "./antinex_core/scripts/standalone_scaler_django.py",
        "./antinex_core/scripts/convert_bottom_rows_to_json.py",
        "./antinex_core/scripts/publish_regression_predict.py",
        "./antinex_core/scripts/update-repos-in-container.sh"
    ],
    use_2to3=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ])
