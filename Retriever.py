from config import get_config
from sentence_transformers import SentenceTransformer
from VectorDatabase import VectorDatabase



class Retriever:
    def __init__(self,config,vector_db):
        
        self.config = config
        self.vector_db = vector_db
        self.model = SentenceTransformer(model_name_or_path=self.config['embedding_model_name'], device=self.config['device'])
        
    def embed_query(self, query):
        """
        Embed the query using the embedding model.
        """
        return self.model.encode(query,convert_to_tensor=True).unsqueeze(0) if query else None
    
    def searc_in_vdb(self, query):
        """
        Search for the top_k most similar documents to the query.
        """
        query_embedding = self.embed_query(query)
        return self.vector_db.search(query_embedding, top_k=self.config['top_k'])

    