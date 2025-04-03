from sentence_transformers import SentenceTransformer
import os
import numpy as np
from spacy.lang.en import English # see https://spacy.io/usage for install instructions

nlp = English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
nlp.add_pipe("sentencizer")

# Create a document instance as an example
doc = nlp("This is a sentence. This another sentence.")
assert len(list(doc.sents)) == 2


""" Creating the vector database class """

class VectorDatabase:
    def __init__(self,config):
        self.config = config
        'initialize the embeddings and documents with empty tensors'
        self.embeddings = torch.empty(0)
        self.documents = []
        self.model = SentenceTransformer(model_name_or_path = config['embedding_model_name'],device = config['device'])

    

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query)
        similarities = [self.cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_k_indices]

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunkify(self):
        chunk_size = self.config['chunk_size']
        chunked_documents = []
        nlp = English()
        nlp.add_pipe("sentencizer")

        for doc in self.documents:
            """ splitting using the nltk library """
            tokens = nlp(doc)
            
            chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
            chunked_documents.extend(chunks)
        self.documents = chunked_documents
        
        

    def add_document(self, document):
        embedding = self.model.encode(document)

        if self.embeddings.size(0) == 0:
            self.embeddings = torch.tensor(embedding).unsqueeze(0)
        else:
            self.embeddings = torch.cat((self.embeddings, torch.tensor(embedding).unsqueeze(0)), dim=0)
        self.embeddings.append(embedding)
        self.documents.append(document)




            """ creationg a chunking function based on the number of tokens """
def chunk_text(text, config):
    """
    Rerads a document and  Splits the text into chunks based on the number of tokens.
    """
    

    config = get_config()
    tokens = word_tokenize(text)
    chunks = []
    chunk_size = config['chunk_size']
    
    for i in range(0, len(tokens), chunk_size):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

