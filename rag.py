from Generator import Generator
from utils import get_config
from VectorDatabase import VectorDatabase
from Retriever import Retriever




class RAG: 
    def __init__(self, config):
        self.config = config
        self.vector_db = VectorDatabase(config)
        self.retriever = Retriever(config, self.vector_db)
        self.generator = Generator(config)
    
    def respond(self, user_prompt):
        """
        Generate a response to the user prompt using RAG.
        """
        # Retrieve relevant documents from the vector database
        retrieved_docs = self.retriever.searc_in_vdb(user_prompt)
        
        # Format and augment the prompt
        augmented_prompt = self.generator.format_and_augment_prompt(user_prompt, retrieved_docs)
        
        # Generate a response using the generator
        response = self.generator.generate_response(augmented_prompt)
        
        return response



