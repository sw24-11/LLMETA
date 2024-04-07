CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

coco_names_path = 'C:/Users/kbh/Code/project2/vision/coco_names.txt'  
imagenet_classes_path = 'C:/Users/kbh/Code/project2/vision/deeper/RelTR/second_classification/imagenet_classes.txt'  

def read_file_to_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file if line.strip()]
    return words

def find_common_words(file1, file2):
    words_file1 = set(read_file_to_list(file1))
    words_file2 = set(read_file_to_list(file2))
    common_words = words_file1.intersection(words_file2)
    
    return common_words

def from_txt_files():

    common_words = find_common_words(coco_names_path, imagenet_classes_path)

    print("Common words (class names) between the two files:")
    with open('common.txt', 'w', encoding='utf-8') as f:
        for word in common_words:
                f.write(word)
                f.write('\n')

def betwwen_list():
    imagenet_classes=set(read_file_to_list(imagenet_classes_path))
    vg_classes=set(CLASSES)
    return list(vg_classes.intersection(imagenet_classes))
print(betwwen_list())