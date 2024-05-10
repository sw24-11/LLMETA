import sys
sys.path.append('./ai/vision/Deeper_RelTR/')

from llm.text_inference import llama2_chain, extract_info
from vision.Deeper_RelTR.img_inference import vision_inference, get_args_parser, argparse

def truncate_text(text, max_length=3072):
    # Split the text into words and truncate to the maximum length allowed
    words = text.split()
    truncated_text = ' '.join(words[:max_length])
    return truncated_text

import re

def extract_before_introduction(text):
    abstract_pattern = re.compile(r'abstract', re.IGNORECASE)
    introduction_pattern = re.compile(r'introduction', re.IGNORECASE)

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

    truncated_paper = extract_before_introduction(research_paper)
    print(truncated_paper)
    llm = llama2_chain(model_paths[0], n_batch=n_batches[2], n_gpu_layers=n_gpu,
                       input_paper=truncated_paper, cb_manager=0)
    llm_chain, prompt = llm.llm_set()
    response = llm_chain.invoke(prompt)
    print(response, '\n\n')
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
            A Study of Method for Metadata Extraction via  
LLM and Scene Graph Generation
Byunghyun Kim, Dayeong Kim, Hyewon Seok, *Dongwook Lee, Seolyoung Jung
Kyungpook National Univ.  *DataStreams
wlfjddlgoqn@knu.ac.kr, rlaekdud@knu.ac.kr, hws2008@knu.ac.kr
*dwlee@datastreams.co.kr, sunflower@knu.ac.kr
Abstract
The significance of metadata in information retrieval and data management is rapidly increasing, particularly in
the realms of academic research and digital content curation. To enhance usability of metadata extraction of
research papers and images with person or specific animals, we propose an advanced method that surpasses
existing techniques 2x accuracy by leveraging Large Language Model (LLM) and Scene Graph Generation (SGG).

Ⅰ. Introduction
As the volume of digital content expands exponentially, traditional
methods of metadata extraction [1] struggle to keep pace, necessi-
tating more sophisticated and scalable approaches. The emergence
of Large Language Models (LLMs) has revolutionized various do-
mains of artificial intelligence, offering unprecedented capabilities
in understanding and summarizing with text. Similarly, advance-
ments in scene graph generation (SGG) [2] have enabled deeper
comprehension of visual content, which is perfect for metadata of
image. We propose even deeper scene graph generation that can
classify nuanced details like, human emotions, races, ages, and
even specific animal breeds [3].
Ⅱ. Text based Method
Traditional methods of metadata extraction from academic papers
[1] often face challenges in accurately and efficiently capturing
key information due to the limitations inherent in simple text pro-
cessing techniques. These challenges are particularly pronounced
when converting PDF files to text as this process can introduce
errors such as misread characters or formatting issues (e.g., ‘hello’
converted to ‘heo’), which conventional text processing tools
struggle to handle. To overcome these difficulties, we employ en-
hanced inference capabilities of Llama2 [4] that can understand
text even when characters are corrupted or converted incorrectly.
Our approach begins with a preprocessing step where PDF files
are converted into text. This text is then inputted into an Llama2
with a specifically crafted prompt designed to extract essential
metadata components such as the title, authors, abstract, references,
and research domain. Llama2’s effectiveness is significantly en-
hanced through prompt-engineering techniques, which refine the
queries made to the model to better suit the nuances of academic
metadata. Additionally, we optimize the extraction process by ad-
justing the temperature and batch size of the model. As illustrated
in Table 1, employing Llama2 significantly enhances metadata ex-
traction accuracy across 246 research papers, achieving nearly a
2.5 times improvement over traditional technologies. This demon-
strates the forte of Llama2 in enhancing academic understanding.

Result
DBLPCheck
Llama2
Authors found
40.24% (99)
98.78% (243)
Authors found +other
33.33% (82)
94.71% (233)
Research Domain
-
97.56% (240)
         Table 1. Compare extraction accuracy

Figure 1. Deeper Scene Graph Architecture

Ⅲ. Image based Method
For image metadata, we extend the utility of traditional tags with
scene graph generation, a powerful tool that can classification and
understanding of the relationships and elements within images. It
is very useful for image searching or even in actively researched
fields [5], [6]. We implement the RelTR [2] model, a state-of-the-
art scene graph generator known for its efficiency in classifying
objects within images. However, the standard implementation of
RelTR includes limited classes related to human subjects (e.g.,
'men', 'man') and animal breeds (e.g., ‘dog’) which can lead to lack
of image explanation. To address this limitation, we integrate
DeepFace [7] technology to classify nuanced human characteris-
tics such as race, age, and emotions. For animal breeds, we utilize
an ImageNet-based detection models [8], [9] that specializes in
identifying specific breeds from images (e.g., ‘Dandie Dinmont’,
‘Tabby’). This approach marginally extends the run-time, but sig-
nificantly enhances the scope of classifiable labels. Also we con-
structed a specialized small test set designed to evaluate our model
capabilities, featuring more specific labels that refine general cat-
egories into distinct entities, such as categorizing 'dog' into 'golden
retriever' in existing test set (e.g., V6).

Figure 2. RelTR and Ours (right)

label
RelTR
Ours
Human (Age×)
12
504
Animal (dog,cat,bird)
3
143
Animal (dog,cat,bird×)
7
204
Others
4
18
          Table 2. Number of classifiable labels
Method
R@50 (V6)
R@50 (V6 +detailed)
RelTR
71.66
33.41

Ours
51.52
64.85

     Table 3. Comparison on V6 and detailed test set
In figure 1, we illustrate our methodology which utilizes an end-
to-end approach through the RelTR checkpoint (i.e., use existing
model path). Here, triplets are detected using the Triplet Decoder.
Subsequent post-processing enables the classification of detected
classes. The post-processing process also includes classifying
which ImageNet classes [3] are included as subclasses among the
existing RelTR classes. Additionally, our post-processing method
allows the use of any ImageNet based models. During further
branches, objects pertinent to human and animal classifications are
selectively routed through the DeepFace and Vision Transformer
(ViT), respectively. This two-stage process ensures that each ele-
ment within the image is accurately tagged, enhancing the depth
and utility of the generated metadata.
As in figure 2, that can classify man to 27 year old happy white
man and dog to golden retriever. This outcome demonstrates that
our more intricate scene graph acquires enhanced capability for
elucidating images. Table 2 highlights the refined classification
features of our methodology across three categories: Human
(Age×), Animal Type, and Others (e.g., ‘vehicle’ to ‘bus’). For
Person class, the model expands from 12 to 504 configurations by
integrating age with 6 races and 7 emotions, providing a detailed
classification of human subjects. The system also possesses the ca-
pability to rectify misclassifications made by the RelTR model.
In Table 3, our model initially underperformed on the V6 dataset
with an R@50 of 51.52% compared to RelTR's 71.66%, due to its
focus on detailed classifications as shown in Figure 2. However,
on V6 test set with more specific labels, the model's performance
significantly improved to 64.85%. This demonstrates its effective-
ness in scenarios requiring precise and granular label classifica-
tions, making it ideal for specialized applications such as targeted
marketing, detailed image-based research, more detailed image
generation with diffusion, and visual question answering.
IV. Conclusion
With Llama2 for text not only enhanced the accuracy metadata ex-
traction but also enriched the granularity of data obtained from ac-
ademic papers. Follow-up research, such as fine-tuning the LLM,
is possible to extract metadata not only from research papers but
also from various texts. And RelTR supplemented by DeepFace
and ImageNet-based models can describe breeds and detailed hu-
man, class post-processing used here can greatly contribute to fu-
ture work, also can leveraged by such diffusion studies or visual
question answering. This holistic approach is further augmented
by the potential for a multi-modal strategy, wherein LLM's lan-
guage understanding capabilities are synergistically combined
with our scene graph model's visual insights. Performance im-
provement can also be expected by leveraging Llama3 [10] and
panoptic scene graph generation [11], or more metrics through ad-
ditional research.
ACKNOWLEDGMENT
"This research was supported by the Korean MSIT (Ministry of Science and ICT),
under the National Program for Excellence in SW)(2021-0-01082) supervised by
the IITP(Institute of Information & communications Technology Planning &
Evaluation)"(2021-0-01082)
REFERENCES
[1] Marinai, S. "Metadata Extraction from PDF Papers," Proc. 10th Int. Conf.
Doc. Anal. Recognit., Barcelona, Spain, July 2009, IEEE,
doi:10.1109/ICDAR.2009.232.
[2] Cong, Y., et al. "RelTR: Relation Transformer for Scene Graph Generation,"
arXiv:2201.11460, Apr. 2023, doi:10.48550/arXiv.2201.11460.
[3] Deng, J., et al. "ImageNet: A Large-Scale Hierarchical Image Database," 2009
IEEE Conf. on Comp. Vision and Pattern Recognition, Miami, FL, June
2009, IEEE Xplore, doi:10.1109/CVPR.2009.5206848.
[4] Touvron, H., et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models,"
arXiv:2307.09288, Jul. 2023, doi:10.48550/arXiv.2307.09288.
[5] Hildebrandt, M., et al. "Scene Graph Reasoning for Visual Question Answer-
ing," arXiv:2007.01072, Jul. 2020, doi:10.48550/arXiv.2007.01072.
[6] Azade, F., et al. "SceneGenie: Scene Graph Guided Diffusion Models for Im-
age Synthesis," arXiv:2304.14573, Apr. 2023,
doi:10.48550/arXiv.2304.14573.
[7] Serengil, S.I., Ozpinar, A. "LightFace: A Hybrid Deep Face Recognition
Framework," 2020 Innovations in Intelligent Systems and Applications Conf.,
Istanbul, Turkey, Oct. 2020, IEEE Xplore,
doi:10.1109/ASYU50717.2020.9259802.
[8] Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale." arXiv:2010.11929 [cs.CV], 3 Jun. 2021,
doi:10.48550/arXiv.2010.11929.
[9] He, K., et al. "Deep Residual Learning for Image Recognition."
arXiv:1512.03385 [cs.CV], 10 Dec. 2015, doi:10.48550/arXiv.1512.03385.
[10] Llama3. (2024). Llama3 GitHub Repository. Available at:
https://github.com/meta-llama/llama3a
[11] Yang, J., Ang, Y. Z., Guo, Z., Zhou, K., Zhang, W., Liu, Z. 'Panoptic Scene
Graph Generation.' arXiv:2207.11247 [cs.CV], 22 Jul 2022,
doi:10.48550/arXiv.2207.11247.
            """
    img_path = './ai/vision/Deeper_RelTR/demo/3.jpg'

    for _ in range(10):
        print(text_inference(research_paper=research_paper))
    #print(image_inference(img_path=img_path))

if __name__=="__main__":
    main()

