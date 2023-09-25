# Feature Visualization Generator UI

[![Python][Python-badge]][Python-url]
[![Pytorch][Pytorch-badge]][Pytorch-url]

A Feature Visualization (FV) generator UI. Utilizes the Gradio web framework in conjunction with the [Lucent](https://github.com/greentfrapp/lucent) (Pytorch [Lucid](https://github.com/tensorflow/lucid) framework. Aimed to help others through experiential learning, allowing them to explore different input parameters and settings and quickly see their effects.

Concepts that are used in the FV generation process such as [Channel Decorrelation and Spatial Decorelation](https://distill.pub/2017/feature-visualization/#d-footnote-8:~:text=the%20training%20data.-,Preconditioning%20and%20Parameterization,-In%20the%20previous) are discussed in the Google Brain Feature Visualization article ([Olah, et al.](https://distill.pub/2017/feature-visualization/)). More can information can be found by searching through recent Explainable Artifical Intelligence (XAI) papers.

Tested layers are Conv2D (Convolutional Layer) and Linear layers. Feel free to experiment, I can not guarantee that they will optimize correctly.
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-code-and-folders">About The Code and Folders</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#website-launch">Website launch</a></li>
      </ul>
    </li>
  </ol>
</details>

### Try it out here! --> https://baxtrax-feature-visualization-generator-ui.hf.space

![image](https://github.com/baxtrax/Feature-Visualization-UI/assets/34373485/226c115e-bb58-40e4-894e-10f5b4282ae5)


<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- ABOUT THE CODE-->
## About The Code and Folders

The python code that runs this website is broken into specific modules to help with readiblity. 

* [models](models)
  * Holds all the model checkpoints used for model loading
* [helpers](helpers)
  * Holds all the different helper methods utilized in the [main.py](main.py) file
* [listeners.py](helpers/listeners.py)
  * All event listener functions used in the main interface
* [models.py](helpers/models.py)
  * Functions that deal with model logic, creation, etc.
* [manipulation.py](helpers/manipulation.py)
  * Functions that deal with data manipulation and logic
* [main.py](main.py)
  * Main interface file. Sets up interface and binds event listeners

> Most complex logic (like feature visualization generation) are in their respective [helpers](helpers) folder.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple steps.

### Prerequisites

You will need to have some sort of python

(Optional)
* Setup a python environment (conda or pyenv) to keep your development space tidy.
* A GPU. This will make generation much, much, quicker

### Installation

1. Clone the repo
   ```bash
   git clone git@github.com:baxtrax/Feature-Visualization-UI.git
   ```
2. Install the required libraries
   ```bash
   pip install -r requirements.txt
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
### Website launch
1. Open a console and run the [main.py](main.py) python file
   ```bash
   python main.py
   ```
2. Click the website link in the console output. It should be on [http://127.0.0.1:7860/](http://127.0.0.1:7860/). This is the default url for gradio.
> The website should try to match your system preferences for dark or light mode. If not, you can specify dark mode and light mode like so: [http://127.0.0.1:7860/?__theme=dark](http://127.0.0.1:7860/?__theme=dark)
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=FFD343
[Python-url]: https://www.python.org/
[Pytorch-badge]: https://img.shields.io/badge/Pytorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/

<!-- Git LFS -->
version https://git-lfs.github.com/spec/v1
oid sha256:8f318c9b980e22dbbcab1dc2c6344e36e3a8c6e9cfa9d7410ed9a798da7b2da1
size 137

