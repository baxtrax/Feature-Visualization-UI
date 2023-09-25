# Feature Visualization Generator UI

[![Python][Python-badge]][Python-url]
[![Pytorch][Pytorch-badge]][Pytorch-url]

A Feature Visualization (FV) generator UI. Utilizes the Gradio web framework in conjunction with the [Lucent](https://github.com/greentfrapp/lucent) (Pytorch [Lucid](https://github.com/tensorflow/lucid) framework. Aimed to help others through experiential learning, allowing them to explore different input parameters and settings and quickly see their effects.

Concepts that are used in the FV generation process such as [Channel Decorrelation and Spatial Decorelation](https://distill.pub/2017/feature-visualization/#d-footnote-8:~:text=the%20training%20data.-,Preconditioning%20and%20Parameterization,-In%20the%20previous) are discussed in the Google Brain Feature Visualization article ([Olah, et al.](https://distill.pub/2017/feature-visualization/))

Tested layers are Conv2D (Convolutional Layer) and Linear layers. Feel free to experiment, I can not guarantee that they will optimize correctly.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-code">About The Code</a>
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
        <li><a href="#html-launch">HTML launch</a></li>
      </ul>
    </li>
  </ol>
</details>

### Try it out here! --> --Insert huggingface space here--

![image](https://github.com/baxtrax/Model-Visualizer/assets/34373485/5c358087-00bb-4bb6-a699-123999ceb367)

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- ABOUT THE CODE-->
## About The Code

The javascript code that runs this website is broken into specific modules to help with readiblity. 

* [canvas.js](scripts/canvas.js)
  * Drawing the lines connecting each button
  * Updating the canvas
  * Button creation / tree traversal
  * Button navigation
  * Button attractions
* [interactions.js](scripts/interactions.js)
  * Model input
  * Model input interaction
  * Orchastraction of the creation and visualization of the new model
* [modelparser.js](scripts/modelparser.js)
  * Parses pytorch print model input string into a tree data structure (Via. regex)
  * Cleaning up string input
* [tree.js](scripts/tree.js)
  * Tree data structure used to store the model and navigate through

> All javascript files are located in the [scripts](scripts) folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple steps.

### Prerequisites

Make sure to have these setup/installed at a minimum
* A modern browser that supports javascript

(Optional)
* A printed pytorch model output to test with
  * This can be achieved by loading up a pytorch model and printing the model with
  ```python
  print(<model variable here>)
  ```
    * I will not be going into detail on how to do this as it is out of the scope of this repo.

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:baxtrax/Model-Visualizer.git
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
### HTML launch
Open the [index.html](index.html) file with a browser or deploy the website with a website deployment software.

I personally used the VScode extension `Live Server` for local development.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=FFD343
[Python-url]: https://www.python.org/
[Pytorch-badge]: https://img.shields.io/badge/Pytorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/


version https://git-lfs.github.com/spec/v1
oid sha256:8f318c9b980e22dbbcab1dc2c6344e36e3a8c6e9cfa9d7410ed9a798da7b2da1
size 137
