from setuptools import setup

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="VUGANEZA Patrice",
    description="A small package for Which Bollywood Celebrity You look like? Deep Learning Project",
    author_email="pvuganeza@gmail.com",
    packages=["src"],
    python_requires=">3.7",
    install_requires=[
        'mtcnn==0.1.0',
        'tensorflow==2.3.1',
        'keras==2.4.3',
        'keras-vggface==0.6',
        'keras_applications==1.0.8',
        'PyYAML',
        'tqdm',
        'scikit-learn',
        'streamlit',
        'bing-image-downloader'
    ]
)
