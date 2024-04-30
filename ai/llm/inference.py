from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import argparse
import re

metadata_extraction_template_2 = """
Given the following Actual Text, extract the title, authors, abstract and research domain. 
and do not say anything else.

Actual Text: {text}
"""

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type=str, default='./models/llama-2-7b-chat.Q2_K.gguf',
                        help="Path of the test image")

def writing_file_pattern():
    with open('expr_pattern.txt', 'w', encoding='utf-8') as f:
        for i in range(10):
            llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu)
            llm_chain, prompt = llm.llm_set()
            response = llm_chain.invoke(prompt)
            f.write(str(response))
            print(i)

def extract_info(text):
    # Regular expression patterns to capture title, authors, abstract, and research domain
    title_pattern = r"Title:\s*(.*?)\s*(?=\n|$)"
    authors_pattern = r"Authors:\s*(.*?)\s*(?=\n|$)"
    abstract_pattern = r"Abstract:\s*(.*?)\s*(?=\n|$)"
    domain_pattern = r"Research Domain:\s*(.*?)\s*(?=\n|$)"
    
    # Search the text for each component using the regex
    title = re.search(title_pattern, text)
    authors = re.search(authors_pattern, text)
    abstract = re.search(abstract_pattern, text)
    domain = re.search(domain_pattern, text)
    
    # Extract the matched text or use a default value if not found
    extracted_info = {
        "Title": title.group(1) if title else "Not provided",
        "Authors": authors.group(1) if authors else "Not provided",
        "Abstract": abstract.group(1) if abstract else "Not provided",
        "Research Domain": domain.group(1) if domain else "Not provided"
    }
    
    return extracted_info

class llama2_chain:

    def __init__(self, model_path, n_gpu_layers, n_batch):
        self.model_path=model_path
        self.n_gpu_layers=n_gpu_layers
        self.n_batch = n_batch
    
    def llm_set(self):
        metadata_extraction_template = """
        Given the following Actual Text, extract the title, authors, abstract and research domain. 
        and do not say anything else.

        Actual Text: {text}

        """

        prompt_template = PromptTemplate(template=metadata_extraction_template, input_variables=["text"])

        #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        callback_manager = CallbackManager([])

        llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            #temperature=0,  
            callback_manager=callback_manager,
            verbose=False,
            n_ctx=3000
        )

        llm_chain = LLMChain(prompt=prompt_template, llm=llm)

        input_text = """
        Autoencoders
        Dor Bank, Noam Koenigstein, Raja Giryes
        Abstract An autoencoder is a specific type of a neural network, which is mainly
        designed to encode the input into a compressed and meaningful representation, and
        then decode it back such that the reconstructed input is similar as possible to the
        original one. This chapter surveys the different types of autoencoders that are mainly
        used today. It also describes various applications and use-cases of autoencoders.
        """

        prompt = prompt_template.format(text=input_text)
        return llm_chain, prompt
    

model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf",]
            #    "C:/Users/kbh/Code/models/llama-2-13b-chat.Q3_K_M.gguf",
            #    "C:/Users/kbh/Code/models/llama-2-13b-chat.Q4_K_M.gguf",
            #    "C:/Users/kbh/Code/models/llama-2-13b-chat.Q5_K_M.gguf",
            #    "C:/Users/kbh/Code/models/llama-2-13b-chat.Q8_0.gguf"]
n_batches = [2000, 3000, 4000]
n_gpu = 50

llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu)
llm_chain, prompt = llm.llm_set()
response = llm_chain.invoke(prompt)
info_dict = extract_info(response['text'])
print(info_dict)


    
