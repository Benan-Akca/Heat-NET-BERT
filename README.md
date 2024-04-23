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


# 3 Running Code
To generate the BERT embeddings and get 2 dimensional UMAP transformation of embeddings run the below code. It will save the umap_embedding.h5 file to output_path

```
bash: python bert_embedding_generator.py --thickness_path DATA_BUCKET/df_thickness.h5 --material_names_path DATA_BUCKET/df_stack.h5 --output_path DATA_BUCKET/umap_embedding.h5

```

To train the model and evaluate results
```
bash: python training_and_evaluation.py
```
The results wil be saved under experiments folder both individually and cumulatively
# 4. Dataset

The dataset comprises pre- and post-heat treatment optical measurements of low-E glass coatings.
Due to proprietary restrictions, it is not publicly available. Contact the corresponding author
for dataset inquiries.
The current dataset in the reposetory is a synthetic dataset to show the data structure of a real data.

# 5. License

The code is available for viewing and academic purposes under strict usage restrictions. Any
form of reproduction, distribution, or commercial use without explicit permission is prohibited.



# 6. Contact Information

- Benan Akca (benanakca@marun.edu.tr)
- Marmara University, Electrical-Electronics Engineering Dept., Istanbul, Turkiye


# 7. Acknowledgments

Supported in part by The Scientific and Technological Research Council of Turkey
(TUBITAK) under Project number 118C067.

