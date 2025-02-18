from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT ='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list of requiremnets
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
setup(
    name='fifa_ml',
    version='0.0.1',
    author='Arpit Agarwal',
    author_email='arpitagarwal20@outlook.com',
    packages = find_packages(),
    # install_requires = ['pandas', 'numpy', 'seaborn']   
    install_requires = get_requirements('requirements.txt')
)