# Machine Unlearning for Face Recognition Demonstration

This repository demonstrates machine unlearning techniques for face recognition systems, implementing and replicating results from the "Bad Teacher Unlearning" methodology.

## Overview

This project implements machine unlearning methods to selectively remove specific identities or data from trained face recognition models without requiring complete retraining. The implementation is based on the paper ["Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher"](https://arxiv.org/abs/2205.08096) (Chundawat et al., AAAI 2023).

## Key Features

- **Student-Teacher Framework**: Uses competent and incompetent teachers to induce selective forgetting
- **Face Recognition Focus**: Specialized for identity-based unlearning in face recognition systems
- **Efficient Unlearning**: Avoid expensive retraining from scratch
- **Evaluation Metrics**: Includes Zero Retrain Forgetting (ZRF) metric for assessment

## Repository Structure

```
├── rocket_unlearn_barebony.ipynb    # Main demonstration notebook
├── race/
│   ├── train.ipynb                  # Training and unlearning implementation
│   └── imdb_crop.tar               # IMDB-WIKI dataset (to be downloaded)
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- tqdm
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/s3975867/machine_learning
cd machine_learning
```

2. Install required dependencies:
```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib
```

## Dataset Setup

1. Download the IMDB-WIKI dataset from the official source:
   - Visit: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
   - Download `imdb_crop.tar`

2. Extract and place the dataset:
```bash
# Place imdb_crop.tar in the /race directory
mv imdb_crop.tar ./race/
cd race
tar -xf imdb_crop.tar
```

## Usage

### 1. Main Demonstration

Run the primary demonstration notebook to see the "Bad Teacher Unlearning" method in action:

```bash
jupyter notebook rocket_unlearn_barebony.ipynb
```

This notebook replicates the key results from the original paper and demonstrates the unlearning process.

### 2. Training and Unlearning

For detailed training and unlearning of face recognition feature extractors:

```bash
cd race
jupyter notebook train.ipynb
```

This notebook includes:
- Face recognition model training
- Implementation of competent/incompetent teacher framework
- Unlearning process for selected identities
- Evaluation using various metrics including ZRF

## Methodology

The implementation follows the "Bad Teacher Unlearning" approach:

1. **Competent Teacher**: Maintains knowledge about data to be retained
2. **Incompetent Teacher**: Designed to "forget" specific target data
3. **Student Model**: Learns selectively from both teachers to achieve targeted forgetting
4. **Evaluation**: Uses Zero Retrain Forgetting (ZRF) metric that doesn't require expensive retraining

## Results

The method demonstrates:
- Effective removal of target identities from face recognition models
- Preservation of performance on retained identities  
- Computational efficiency compared to retraining from scratch
- Robustness against membership inference attacks

## License

Please refer to the [original paper](https://github.com/vikram2000b/bad-teaching-unlearning/tree/main)'s license and the IMDB-WIKI dataset license for usage terms.

## Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes.