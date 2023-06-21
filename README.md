# Handwriting OCR

## Introduction

After a long period of radio silence on GitHub from me, I have now made a neural network for computer science, my very first machine learning project (not counting my genetic algorithm project, because I'm unsure whether that counts as machine learning). I thought it would be useful to have a program that could read my handwriting (one of the only things on earth that could), so any notes I make in classes could automatically be added to a notes program like notion without much work of my own. I've always wanted to try something within machine learning, and this was a good excuse to do so.

It is very inaccurate, and that is most likely just because of the `test.py` file. In order to train a model, you must input the size of the images you will be training it on, and this must correspond with the image size of the image you will use to test the model, which means you are only really able to accurately predict individual letters. To solve this, I attempted to make a sort of rolling window type system, which moves the window 28 pixels to the right until the next set of black pixels is in view of the window, and you have an array of individual letters. Obviously this doesn't work well at all, but I needed to hand something in for computer science. This is a project I would like to finish at some time in the future, as it is an interesting project to say the least.

## Usage

To use this, please follow the next steps:

1. Download the Kaggle A-Z Dataset (the CSV version), and extract the CSV file into the same directory as the Python files. You can find the dataset here: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
2. Install the necessary libraries: `pip install keras scikit-learn pandas matplotlib numpy cv2`
3. Run train.py: `python train.py`
4. Upload the image with handwritten text (only maximum of 1 line) in the same directory and name it tekst.png
5. Run test.py: `python test.py`