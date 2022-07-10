from setuptools import setup

setup(
    name="shogiraffe",
    version="0.6",
    description="A shogi AI, interface and board recognition project",
    url="https://github.com/Theo-Cheynel/shogiraffe",
    author="Théo Cheynel",
    author_email="theo.cheynel@gmail.com",
    packages=[
        "shogiraffe",
        "shogiraffe.interface",
        "shogiraffe.strategy",
        "shogiraffe.image_recognition",
        "shogiraffe.web_scraping",
        "shogiraffe.environment",
    ],
    package_data={"": ["data/*"]},
    install_requires=["numpy", "torch", "python-shogi", "ddt"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
    ],
)
