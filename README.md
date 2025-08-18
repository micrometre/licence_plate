# ANPR Image Preprocessing

This project demonstrates the key image preprocessing steps for Automatic Number Plate Recognition (ANPR) using OpenCV and Python, based on the article [ANPR Series Part 1: Advanced Image Preprocessing Techniques](https://henok.cloud/articles/anpr-part-1-image-preprocessing/).


## Features
- Step-by-step Jupyter notebooks for ANPR:
	- **Part 1:** Image preprocessing (grayscale, noise reduction, blurring, edge detection, morphology, contour detection)
	- **Part 2:** Plate detection engine (contour filtering, geometric validation, edge density, candidate visualization)
- Visualizes each processing step for easy understanding

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or VS Code with Jupyter extension
- OpenCV, NumPy, Matplotlib


### Create and Activate a Virtual Environment

It is recommended to use a Python virtual environment to manage dependencies:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment (Linux/macOS)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

### Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install opencv-python numpy matplotlib
```


### Usage
1. Place your sample image in the `sample_images/` directory (or update the path in the notebooks).
2. Open `anpr_image_preprocessing.ipynb` to run the preprocessing pipeline (Part 1).
3. Open `anpr_plate_detection.ipynb` to run the plate detection pipeline (Part 2). Make sure to run preprocessing first or provide a preprocessed image.
4. Run each cell to see the full ANPR workflow in action.


## Reference
- [ANPR Series Part 1: Advanced Image Preprocessing Techniques](https://henok.cloud/articles/anpr-part-1-image-preprocessing/)
- [ANPR Series Part 2: Advanced Plate Detection Engine](https://henok.cloud/articles/anpr-part-2-plate-detection/)

## License
This project is for educational purposes. See the original article for more details.
