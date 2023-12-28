#!/usr/bin/env python3

# [Set up Google cloud credentials](https://cloud.google.com/docs/authentication/external/set-up-adc)
#export CLOUDSDK_PYTHON=python3
#brew install wget
#wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-darwin-x86_64.tar.gz
#tar zxvf google-cloud-cli-458.0.1-darwin-x86_64.tar.gz
#./google-cloud-sdk/install.sh
#./google-cloud-sdk/bin/gcloud init

# [Using Google Palm with Llamaindex.](https://docs.llamaindex.ai/en/stable/examples/llm/palm.html)
#Google project ID [concise-complex-241222]
#Google cloud API []
#Activate the VertexAI API when prompted.

#gcloud auth application-default login
#pip install google-generativeai

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import ServiceContext
from llama_index.llms import PaLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader

service_context = ServiceContext.from_defaults(llm=PaLM(api_key=""))

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print(response)
