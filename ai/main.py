import sys
sys.path.append('./ai/vision/Deeper_RelTR/')

from llm.text_inference import llama2_chain, extract_info
from vision.Deeper_RelTR.img_inference import vision_inference, get_args_parser, argparse

def text_inference(research_paper):
    model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf"]
    n_batches = [2000, 3000, 4000]
    n_gpu = 50

    llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu, input_paper=research_paper)
    llm_chain, prompt = llm.llm_set()
    response = llm_chain.invoke(prompt)
    info_dict = extract_info(response['text'])

    return info_dict
    #print(info_dict)

def image_inference(img_path):
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser(img_path=img_path)])
    args = parser.parse_args()
    caption, triplet_graph = vision_inference(args)
    print(caption)

    return triplet_graph

def main():
    research_paper = """
            Autoencoders
            Dor Bank, Noam Koenigstein, Raja Giryes
            Abstract An autoencoder is a specific type of a neural network, which is mainly
            designed to encode the input into a compressed and meaningful representation, and
            then decode it back such that the reconstructed input is similar as possible to the
            original one. This chapter surveys the different types of autoencoders that are mainly
            used today. It also describes various applications and use-cases of autoencoders.
            """
    img_path = './ai/vision/Deeper_RelTR/demo/3.jpg'

    print(text_inference(research_paper=research_paper))
    print(image_inference(img_path=img_path))

if __name__=="__main__":
    main()

