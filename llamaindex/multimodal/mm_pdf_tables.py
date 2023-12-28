#!/usr/bin/env python
# coding: utf-8

# # Multi-Modal on PDF's with tables.

# Uses Qdrant vector storage
# docker pull qdrant/qdrant
# docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

# install prereqs
get_ipython().system('pip install llama-index qdrant_client pyMuPDF tools frontend git+https://github.com/openai/CLIP.git easyocr')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms

from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import fitz

device = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_API_TOKEN = "sk-"
openai.api_key = OPENAI_API_TOKEN

get_ipython().system('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "llama2.pdf"')

pdf_file = "llama2.pdf"

# Split the base name and extension
output_directory_path, _ = os.path.splitext(pdf_file)

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# Open the PDF file
pdf_document = fitz.open(pdf_file)

print("Converting PDFs to images...")

# Iterate through each page and convert to an image
for page_number in range(pdf_document.page_count):
    # Get the page
    page = pdf_document[page_number]

    # Convert the page to an image
    pix = page.get_pixmap()

    # Create a Pillow Image object from the pixmap
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image
    image.save(f"./{output_directory_path}/page_{page_number + 1}.png")

# Close the PDF file
pdf_document.close()

from PIL import Image
import matplotlib.pyplot as plt
import os

image_paths = []
for img_path in os.listdir("./llama2"):
    image_paths.append(str(os.path.join("./llama2", img_path)))


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

#plot_images(image_paths[9:12])

import qdrant_client
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index import VectorStoreIndex, StorageContext
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.schema import ImageDocument

from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
)

# Read the images
documents_images = SimpleDirectoryReader("./llama2/").load_data()

print("Create the vector store... ")

# Create a Docker Qdrant vector store
client = qdrant_client.QdrantClient(url="http://localhost:6333")

from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

text_store = client.recreate_collection(
    collection_name="text_collection",
    vectors_config=VectorParams(
            size=1536,  # openai vector size
            distance=models.Distance.COSINE,
        )
)

image_store = client.recreate_collection(
    collection_name="image_collection",
    vectors_config=VectorParams(
            size=1536,  # openai vector size
            distance=models.Distance.COSINE,
        )
)

print("Create the image and text embeddings... ")

# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents_images,
    vector_storage=text_store,
    image_store=image_store
)

retriever_engine = index.as_retriever(image_similarity_top_k=2)

from llama_index.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)

query = "Compare llama2 to llama1?"
assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)

# retrieve for the query using text to image retrieval
retrieval_results = retriever_engine.text_to_image_retrieve(query)

retrieved_images = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_images.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

# test plot_images and text_to_image_retrieve
#plot_images(retrieved_images)

# Modified utils from https://huggingface.co/spaces/nielsr/tatr-demo

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def detect_and_crop_save_table(
    file_path, cropped_table_directory="./table_images/"
):
    image = Image.open(file_path)

    filename, _ = os.path.splitext(file_path.split("/")[-1])

    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)

    # prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    print(f"number of tables detected {len(detected_tables)}")

    for idx in range(len(detected_tables)):
        #   # crop detected table out of image
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        cropped_table.save(f"./{cropped_table_directory}/{filename}_{idx}.png")


print("Create tables from retrieved images and sent to transformer... ")

for file_path in retrieved_images:
    detect_and_crop_save_table(file_path)

# Read the cropped tables
image_documents = SimpleDirectoryReader("./table_images/").load_data()

response = openai_mm_llm.complete(
    prompt="Compare llama2 with llama1?",
    image_documents=image_documents
)

print(response)
#plot_images(retrieved_images)
