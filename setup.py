from setuptools import setup

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]
    
setup(
    name='CFN ETEM',
    version='0.1',
    packages=['cfntem'],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu',
    author_email='xiaqu@bnl.gov',
    description='CFN ETEM Analytics',
    python_requires='>=3.8',
    install_requires=required_list,
    entry_points={
        "console_scripts": [
            "read_K2 = cfntem.cmd.read_K2:main"
        ]
    },
    scripts=[
    ]
)

