from setuptools import find_packages
from setuptools import setup

# https://cloud.google.com/ai-platform/training/docs/runtime-version-list#2.2
REQUIRED_PACKAGES = [
    'python-dotenv==0.15.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='generic trainer package'
)
