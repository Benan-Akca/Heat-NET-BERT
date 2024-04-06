# AI-Enhanced Design and Manufacture Process for Low-E Glass Coating Project README

# 1. Overview

This repository contains the code and models developed during our research on
AI-enhanced design and manufacture processes for low-emissivity (low-E) glass
coatings. Utilizing a combination of large language models (BERT) and Convolutional
Neural Networks (CNNs), this AI-based method accurately predicts the optical behavior
of glass coatings post-heat treatment, streamlining the development process for
low-E glass coatings and reducing reliance on traditional trial-and-error methods.


# 2. Installation

Prerequisites:
- Python 3.8+
- PyTorch 1.8+
- Transformers library
- See `requirements.txt` for other dependencies.

To install the required Python packages, execute in the terminal:
bash: pip install -r requirements.txt


# 3. Usage
## 3.1 Training the Model

To train the model with the provided dataset, execute:
bash: python train_model.py --dataset_path /path/to/your/dataset


## 3.2 Making Predictions

To make predictions using a pre-trained model, execute:
```
bash: python predict.py --model_path /path/to/your/model --input_data /path/to/your/input_data
```

# 4. Dataset

The dataset comprises pre- and post-heat treatment optical measurements of low-E glass coatings.
Due to proprietary restrictions, it is not publicly available. Contact the corresponding author
for dataset inquiries.


# 5. Contributing

Contributions to improve the models and code are welcome. Fork the repository and submit pull
requests for minor changes. For significant modifications, please open an issue first to discuss
what you would like to change.


# 6. License

The code is available for viewing and academic purposes under strict usage restrictions. Any
form of reproduction, distribution, or commercial use without explicit permission is prohibited.


# 7. Citation

If utilizing this work in research or for academic purposes, please cite as follows:
@article{Akca2024,
  title={AI-Enhanced Design and Manufacture Process for Low-E Glass Coating},
  author={Benan Akca et al.},
  journal={Nature Journal},
  year={2024}
}


# 8. Contact Information

- Benan Akca (benanakca@marun.edu.tr)
- Marmara University, Electrical-Electronics Engineering Dept., Istanbul, Turkiye


# 9. Acknowledgments

Supported in part by The Scientific and Technological Research Council of Turkey
(TUBITAK) under Project number 118C067.

