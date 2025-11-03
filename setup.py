import setuptools

setuptools.setup(
    
    name='audio-speech-to-sign-language-converter',
    version='0.1.0',
    description='Python project',
    author='abinesh',
    author_email='05abinesh06@gmail.com',
    url='https://github.com/Abineshgiriraj/AI-Powered-ISL-Communication-Assistant-for-Inclusive-Interaction',
    packages=setuptools.find_packages(),
    setup_requires=['nltk', 'joblib','click','regex','sqlparse','setuptools'],
)