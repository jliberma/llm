#!/usr/bin/env python
# coding: utf-8

# # Chroma Multi-Modal Demo with LlamaIndex
# 
# Install Chroma with:
# 
# ```sh
# pip install chromadb
# ```

#get_ipython().system('pip install llama-index')

# #### Creating a Chroma Index

#get_ipython().system('pip install llama-index chromadb --quiet')
#get_ipython().system('pip install chromadb==0.4.17')
#get_ipython().system('pip install sentence-transformers')
#get_ipython().system('pip install pydantic==1.10.11')
#get_ipython().system('pip install open-clip-torch')

#get_ipython().system('cd examples/multimodal/chroma')

# import
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, Image, display
import chromadb

# set up OpenAI
import os
import openai

OPENAI_API_KEY = "sk-"
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ## Download Images and Texts from Wikipedia

import requests

def get_wikipedia_images(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|dimensions|mime",
            "generator": "images",
            "gimlimit": "50",
        },
    ).json()
    image_urls = []
    for page in response["query"]["pages"].values():
        if page["imageinfo"][0]["url"].endswith(".jpg") or page["imageinfo"][
            0
        ]["url"].endswith(".png"):
            image_urls.append(page["imageinfo"][0]["url"])
    return image_urls

from pathlib import Path
import urllib.request

image_uuid = 0
MAX_IMAGES_PER_WIKI = 20

wiki_titles = {
#    "Tesla Model X",
#    "Pablo Picasso",
#    "Rivian",
    "The Lord of the Rings",
    "The Matrix",
#    "The Simpsons",
}

data_path = Path("mixed_wiki")
if not data_path.exists():
    Path.mkdir(data_path)

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

    images_per_wiki = 0
    list_img_urls = get_wikipedia_images(title)
    #print(list_img_urls)

    for url in list_img_urls:
        if url.endswith(".jpg") or url.endswith(".png"):
            image_uuid += 1
            #image_file_name = title + "_" + url.split("/")[-1]
            
            urllib.request.urlretrieve(
                url, data_path / f"{image_uuid}.jpg"
            )
            images_per_wiki += 1
            # Limit the number of images downloaded per wiki page to 15
            print(title," ",images_per_wiki)
            if images_per_wiki > MAX_IMAGES_PER_WIKI:
                break

get_ipython().system(' ls mixed_wiki')

# ## Set the embedding model

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# set defalut text and image embedding functions
embedding_function = OpenCLIPEmbeddingFunction()

# ## Build Chroma Multi-Modal Index with LlamaIndex

from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext
from chromadb.utils.data_loaders import ImageLoader

image_loader = ImageLoader()

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# load documents
documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# ## Retrieve results from Multi-Modal Index

retriever = index.as_retriever(similarity_top_k=50)
retrieval_results = retriever.retrieve("Matrix posters")

# print(retrieval_results)
from llama_index.schema import ImageNode
from llama_index.response.notebook_utils import (
    display_source_node,
    display_image_uris,
)

image_results = []
MAX_RES = 5
cnt = 0
for r in retrieval_results:
    if isinstance(r.node, ImageNode):
        image_results.append(r.node.metadata["file_path"])
    else:
        if cnt < MAX_RES:
            display_source_node(r)
        cnt += 1

display_image_uris(image_results, [3, 3], top_k=2)

chroma_collection = chroma_client.delete_collection("multimodal_collection")
get_ipython().system(' rm mixed_wiki/*')
