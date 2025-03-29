# import torch
# x =torch.empty(0)

# print(x)
# print(x.shape)  

# y = 1

# if x.size(0) == 0:
#     x = torch.tensor(y).unsqueeze(0)
# else:
#     x = torch.cat((x, torch.tensor(y).unsqueeze(0)), dim=0)

# print(x)  
# print(x.shape)  

# if x.size(0) == 0:
#     x = torch.tensor(y).unsqueeze(0)
# else:
#     x = torch.cat((x, torch.tensor(y).unsqueeze(0)), dim=0)

# print(x)    
# print(x.shape)

from spacy.lang.en import English # see https://spacy.io/usage for install instructions

nlp = English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
nlp.add_pipe("sentencizer")

# Create a document instance as an example
doc = nlp("This is a sentence. This another sentence.")


# Access the sentences of the document
print(doc.sents)