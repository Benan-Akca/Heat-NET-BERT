import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import umap
import argparse


class DataHandler:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.data = {}

    def load_data(self):
        self.data["layer_thickness"] = joblib.load(self.data_paths["thickness"])
        self.data["material_names"] = joblib.load(self.data_paths["material_names"])

    def get_data(self, key):
        return self.data.get(key, None)


class TextProcessor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_sentence(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def preprocess(self, sentence):
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokens_tensor, segments_tensors


class EmbeddingGenerator:
    def __init__(self, model_name="bert-base-uncased"):
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.text_processor = TextProcessor(model_name)

    def generate_embeddings(self, sentences, thickness_data):
        embeddings = []
        for index, sentence in enumerate(sentences):
            tokens_tensor, segments_tensors = self.text_processor.preprocess(sentence)
            thickness_array = thickness_data.iloc[index, :].values
            thickness_repet = self.get_repet_thickness(sentence, thickness_array)
            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            embed = hidden_states[12][0][1:-1]
            embed_array = np.array(embed)
            new_array = embed_array * thickness_repet
            mean_embedding = new_array.mean(axis=0)
            embeddings.append(mean_embedding)
        return np.array(embeddings)

    def get_repet_thickness(self, sentence, thickness_array):
        material_len = [
            len(self.text_processor.tokenize_sentence(i)) for i in sentence.split()
        ]
        thickness_list = [[str(i)] for i in thickness_array.tolist()]
        thc_list_raw = []
        for index, length in enumerate(material_len):
            thc_list_raw.append(length * thickness_list[index])
        thc_list = sum(thc_list_raw, [])
        repet_thick_list = [float(i) for i in thc_list]
        return np.array(repet_thick_list).reshape(-1, 1)


class UMAPEmbedder:
    def __init__(self):
        self.reducer = umap.UMAP()

    def fit_transform(self, data):
        scaled_data = StandardScaler().fit_transform(data)
        embedding = self.reducer.fit_transform(scaled_data)
        return embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and plot UMAP embeddings from BERT model outputs."
    )
    parser.add_argument(
        "--thickness_path",
        type=str,
        required=True,
        help="Path to the thickness data file",
    )
    parser.add_argument(
        "--material_names_path",
        type=str,
        required=True,
        help="Path to the material names CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path to save the umap embeddings",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_paths = {
        "thickness": args.thickness_path,
        "material_names": args.material_names_path,
    }

    data_handler = DataHandler(data_paths)
    data_handler.load_data()

    material_names_df = data_handler.get_data("material_names")
    sentences = material_names_df["Coating Stack"].values.tolist()

    layer_thickness_df = data_handler.get_data("layer_thickness")

    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(sentences, layer_thickness_df)

    umapper = UMAPEmbedder()
    umap_embeddings = umapper.fit_transform(embeddings)

    umap_df = pd.DataFrame(
        umap_embeddings, columns=["dim1", "dim2"], index=layer_thickness_df.index
    )
    joblib.dump(umap_df, args.output_path)
    print("The BERT embeddings with UMAP transformation have been saved successfully!")
