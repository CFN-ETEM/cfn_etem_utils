from setuptools import setup

setup(
    name='CFN ETEM,
    version='0.1',
    packages=[''],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu',
    author_email='xiaqu@bnl.gov',
    description='CFN ETEM Analytics',
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "read_K2 = cfntem.cmd.read_K2:main"
        ]
    },
    scripts=[
    ]
)

