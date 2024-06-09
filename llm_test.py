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

# def writing_file_pattern():
#     with open('expr_pattern.txt', 'w', encoding='utf-8') as f:
#         for i in range(10):
#             llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu)
#             llm_chain, prompt = llm.llm_set()
#             response = llm_chain.invoke(prompt)
#             f.write(str(response))
#             print(i)

def extract_info(text):
    title_pattern = r"Title:\s*(.*?)\s*(?=\n|$)"
    authors_pattern = r"Author(?:s)?(?:\(s\))?:\s*(.*?)\s*(?=\n|$)"  # Updated pattern
    abstract_pattern = r"Abstract:\s*(.*?)\s*(?=\n|$)"
    domain_pattern = r"Research Domain:\s*(.*?)\s*(?=\n|$)"
    
    title = re.search(title_pattern, text)
    authors = re.search(authors_pattern, text)
    abstract = re.search(abstract_pattern, text)
    domain = re.search(domain_pattern, text)
    
    extracted_info = {
        "Title": title.group(1) if title else "Not provided",
        "Authors": authors.group(1) if authors else "Not provided",
        "Abstract": abstract.group(1) if abstract else "Not provided",
        "Research Domain": domain.group(1) if domain else "Not provided"
    }
    
    return extracted_info


class llama2_chain:

    def __init__(self, model_path, n_gpu_layers, n_batch, input_paper, cb_manager):
        self.model_path=model_path
        self.n_gpu_layers=n_gpu_layers
        self.n_batch = n_batch
        self.input_paper = input_paper
        self.cb_manager = cb_manager
    
    def llm_set(self):
        metadata_extraction_template = """

        Answer with this format from research paper and do not say anything else. 

        Title:
        Authors:
        Abstract:
        Research Domain:

        research paper is {text}
        """

        prompt_template = PromptTemplate(template=metadata_extraction_template, input_variables=["text"])
        callback_manager = [CallbackManager([StreamingStdOutCallbackHandler()]), CallbackManager([])]
                
        llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            temperature=0,  
            callback_manager=callback_manager[self.cb_manager],
            verbose=False,
            n_ctx=3000
        )

        llm_chain = LLMChain(prompt=prompt_template, llm=llm)

        prompt = prompt_template.format(text=self.input_paper)
        return llm_chain, prompt

def truncate_text(text, max_length=3072):
    words = text.split()
    truncated_text = ' '.join(words[:max_length])
    return truncated_text

import re
import fitz

def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ''
    for page in document:
        text += page.get_text("text")
    return text

def extract_before_introduction(text):
    abstract_pattern = re.compile(r'abstract', re.IGNORECASE)
    introduction_pattern = re.compile(r'introduction', re.IGNORECASE)
    reference_pattern = re.compile(r'reference', re.IGNORECASE)

    abstract_match = abstract_pattern.search(text)
    if not abstract_match:
        return "Abstract section not found"

    introduction_match = introduction_pattern.search(text, abstract_match.end())
    if not introduction_match:
        return text[0:abstract_match.start()]

    return text[0:introduction_match.start()]

def text_inference(research_paper):
    model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf"]
    n_batches = [2000, 3000, 4000]
    n_gpu = 50

    # Truncate text to fit within the model's token limit
    truncated_paper = extract_before_introduction(research_paper)

    llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu,
                       input_paper=truncated_paper, cb_manager=0)
    llm_chain, prompt = llm.llm_set()
    response = llm_chain.invoke(prompt)
    info_dict = extract_info(response['text'])
    return info_dict

text_content = extract_text_from_pdf("C:/Users/kbh/Code/project2/vision/LLMETA/ai/uploads/paper.pdf")
metadata = text_inference(text_content)
merged_list = [{"key": key, "value": value} for key, value in metadata.items()]
print(merged_list)
