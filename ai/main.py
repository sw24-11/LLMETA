from llm.inference import llama2_chain, extract_info
from vision.Deeper_RelTR.inference_all import vision_inference
model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf"]
n_batches = [2000, 3000, 4000]
n_gpu = 50

research_paper = """
        Autoencoders
        Dor Bank, Noam Koenigstein, Raja Giryes
        Abstract An autoencoder is a specific type of a neural network, which is mainly
        designed to encode the input into a compressed and meaningful representation, and
        then decode it back such that the reconstructed input is similar as possible to the
        original one. This chapter surveys the different types of autoencoders that are mainly
        used today. It also describes various applications and use-cases of autoencoders.
        """

llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu, input_paper=research_paper)
llm_chain, prompt = llm.llm_set()
response = llm_chain.invoke(prompt)
info_dict = extract_info(response['text'])
print(info_dict)