# Vector Space Retrieval Model

## Overview

This program is a search engine that searches for relevant documents in a given collection based on a query. 

The search engine uses the vector space retrieval model for specifying how to rank the documents. The collection of documents is stored in an XML file, which is preprocessed and indexed using TF-IDF scores. The core of the search engine is an index that weights terms according to the tf.idf weighting scheme.


## Project Structure 

    ├── LICENSE
    ├── README.md
    ├── code
    │   └── searchEngine.py
    ├── data
    │   └── nytsmall.xml
    ├── report
    │   └── SearchEngine_doc.pdf
    ├── requirements.txt
    └── utils
        └── stemming


## Usage

1. To run the program, create a SearchEngine object by calling the constructor with the name of the collection and a boolean value indicating whether to create a new index or load an existing one. 

2. Then, use the executeQueryConsole method on the object. When the program is working, provide the query by typing in the words and pressing “ Enter”. The program will be running until an empty query is prompted.


## Dependencies
- Python 3.x
- Required Python packages (install using pip install -r requirements.txt)