# Live_Sudoku_Solver

## Introduction
This is a Computer Vision Application that solves a 9x9 Sudoku Puzzle. The puzzle is solved by using a Deep Learning Neural Network model to predict the digits in the image. The digits are extracted and solved using Backtracking Algorithm, which is a popular method of solving a sudoku puzzle. The newly found digits are then placed in the empty cells of the puzzle. The main feature included in this application is the ability to solve the puzzle by reading in live video of the puzzle through the Computer's webcam. 
## Dependency
- Python3, Ubuntu 18.04 or WindowsOS
- OpenCV, Tensorflow, Keras, Pillow
- To install the required packages, run pip install -r requirements.txt
## Dataset for Training Model
The CNN model used for this application is the Keras MNIST Handwritten Digit Classification.
## Usage 
- First, clone the repository with git clone and enter the cloned folder.
- Create a virtual environment containing the requirement libraries from the requirements.txt file and activate it.
- Run model.py to train the model on the dataset
- Facing your webcam with an image of the puzzle, enter python live.py in your terminal and the puzzle is solved
