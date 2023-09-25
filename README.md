# Feature Visualization Generator UI

[![HTML][Html-badge]][Html-url]
[![CSS][CSS-badge]][CSS-url]
[![JS][JS-badge]][JS-url]

Easy to use website that dynamically gives a view of a model. Additonally can show the feature visualizations of a network when hovering over layers that have feature visualizations generated.

The website will default to a [Resnet18 (Kaiming He et al.)](https://arxiv.org/abs/1512.03385) network that has been modified to work with the [CIFAR10 (Alex Krizhevsky)](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) dataset. **Additionally all layers that are `Conv2d` layers will show a feature visualization on hover.**

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

### Try it out here! --> https://baxtrax.github.io/Model-Visualizer/

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
[HTML-badge]: https://img.shields.io/badge/HTML-E34F26.svg?style=for-the-badge&logo=html5&logoColor=white
[HTML-url]: https://developer.mozilla.org/en-US/docs/Web/HTML


version https://git-lfs.github.com/spec/v1
oid sha256:8f318c9b980e22dbbcab1dc2c6344e36e3a8c6e9cfa9d7410ed9a798da7b2da1
size 137
