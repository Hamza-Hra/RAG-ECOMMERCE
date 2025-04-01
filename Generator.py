from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from config import get_config
from transformers import BitsAndBytesConfig


# Login to Hugging Face Hub
login("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")  # Replace with your Hugging Face token


""" Wrapping the model in a class """

class Generator:
    def __init__(self, config):
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['generator_model_name'])
        if config['use_quantization_config']:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            self.quantization_config = None
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config['generator_model_name'],
            quantization_config=self.quantization_config,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16,
            attn_implementation=config['attn_implementation']
        )

        self.model.to(config['device'])

    def generate_response(self,augmented_prompt):
        """
        Generate text using the augmented prompt.
        """
        input_ids = self.tokenizer(augmented_prompt, return_tensors="pt").to(self.config['device'])
        outputs = self.model.generate(**input_ids,max_new_tokens=self.config['max_new_tokens'], do_sample=True, temperature=1)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        'Removinbg the prompt from the output'
        output = decoded_output.replace(augmented_prompt,'').strip()
        'Removing the answer prefix'
        output = output.replace("Answer:", "").strip()



        return output

    def format_and_augment_prompt(self, user_prompt, retrieved_docs):
        """
        Format the prompt by adding the user prompt and retrieved documents.
        """
    
        # Join retrieved documents into a single string
        context = "\n".join(retrieved_docs)
    
        # Format the prompt to explicitly guide the model
        prompt = f"""You are an AI assistant that provides direct answers to user questions. 
        Use the following context to answer the query concisely. Always format your response as:
        "The answer is <answer>."
        
        Context:
        {context}
        
        User query: {user_prompt}
        Your response:"""

        dialog_template = [
            {
                "role": "user",
                "content": prompt
            },
        ]
    
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialog_template,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt



'testing the generator'

if __name__ == "__main__":
    config = get_config()
    generator = Generator(config)
    user_prompt = "What is the capital of France?"
    retrieved_docs = ["Paris is the capital of France.", "France is a country in Europe."]
    augmented_prompt = generator.format_and_augment_prompt(user_prompt, retrieved_docs)
    response = generator.generate_response(augmented_prompt)
    print(response)
            

   



