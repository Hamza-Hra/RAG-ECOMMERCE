from sentence_transformers import SentenceTransformer
import os
import numpy as np
from spacy.lang.en import English # see https://spacy.io/usage for install instructions




""" Creating the vector database class """

class VectorDatabase:
    def __init__(self,config):
        self.config = config
        'initialize the embeddings and documents with empty tensors'
        self.embeddings = []
        self.documents = []
        #self.vector_db = []
        self.model = SentenceTransformer(model_name_or_path = config['embedding_model_name'],device = config['device'])

    

    def search(self, query_embedding, top_k=5):

        similarities = [self.cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_k_indices]

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunkify_and_add_doc(self,file_path):
        """
        Chunkify the documents into smaller pieces.
        """
        number_sentences = self.config['chunk_size']
        chunked_document = []

        if not os.path.exists(file_oath):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        else:
            print(f"File {file_path} exists.")
            with open(file_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
        
        nlp = English()
        nlp.add_pipe("sentencizer")

        document_sentences = nlp(document_content)
        sentences = []
        for sent in document_sentences.sents:
            sentences.append(sent.text)
        
        'creating the chunks'
        for i in range(0, len(sentences), number_sentences):
            chunk = [' '.join(sentences[i:i + number_sentences])]
            self.documents.append(chunk)
            self.embeddings.append(self.model.encode(chunk, convert_to_tensor=True).unsqueeze(0))
        
            



