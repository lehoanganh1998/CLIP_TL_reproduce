import yaml
from torchvision.datasets import *
import clip
import torch
import os
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#! DEVICE DEFINE
device = "cuda" if torch.cuda.is_available() else "cpu"
#! FUNCTION DEFINITION
# Zeroshot
def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

# Calculate Accuracy
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]        

# dataset loader   
def dataset_loader(dataset_name):
    # load dataset
    if dataset_name == "CIFAR10":
        images = CIFAR10(root=os.path.expanduser("Dataset/"), download=True, train=False,transform=preprocess)
        
    elif dataset_name == "Birdsnap":
        images = None    

    elif dataset_name == "CIFAR100":
        images = CIFAR100(root=os.path.expanduser("Dataset/"), download=True, train=False,transform=preprocess)

    elif dataset_name == "CLEVRCounts":
        images = None

    elif dataset_name == "Caltech101":
        # images = Caltech101(root=os.path.expanduser("Dataset/"), download=True, transform=preprocess, target_type="category")
        images = None
    
    elif dataset_name == "Country211":
        images = None

    elif dataset_name == "DescribableTextures":
        images = DTD(root=os.path.expanduser("Dataset/"), download=True, split="test", transform=preprocess)

    elif dataset_name == "EuroSAT":
        # images = EuroSAT(root=os.path.expanduser("Dataset/"), download=True, transform=preprocess)
        images = None
        
    elif dataset_name == "FGVCAircraft":
        images = FGVCAircraft(root=os.path.expanduser("Dataset/"), download=True, split="test", transform=preprocess)

    elif dataset_name == "FacialEmotionRecognition2013":
        images = None

    elif dataset_name == "Flowers102":
        images = Flowers102(root=os.path.expanduser("Dataset/"),split="test", download=True, transform=preprocess)

    elif dataset_name == "Food101":
        images = Food101(root=os.path.expanduser("Dataset/"), download=True, split="test", transform=preprocess)

    elif dataset_name == "GTSRB":
        images = GTSRB(root=os.path.expanduser("Dataset/"), download=True, split="test",transform=preprocess)

    elif dataset_name == "HatefulMemes":
        images = None

    elif dataset_name == "KITTI":
        # images = Kitti(root=os.path.expanduser("Dataset/"), train=False, download=True, transform=preprocess)
        images = None
        
    elif dataset_name == "Kinetics700":
        # images = Kinetics(root=os.path.expanduser("Dataset/"),num_classes=700, split="test", transform=preprocess,download=True)
        images = None
        
    elif dataset_name == "MNIST":
        images = MNIST(root=os.path.expanduser("Dataset/"), download=True, train=False,transform=preprocess)

    elif dataset_name == "OxfordPets":
        # images = OxfordIIITPet(root=os.path.expanduser("Dataset/"), split="test", transform=preprocess, download=True)
        images = None
        
    elif dataset_name == "PascalVOC2007":
        # images = VOCDetection(root=os.path.expanduser("Dataset/"), year="2007", image_set="test",download=True, transform=preprocess)
        images = None
        
    elif dataset_name == "PatchCamelyon":
        images = None

    elif dataset_name == "RESISC45":
        images = None

    elif dataset_name == "SST2":
        images = None

    elif dataset_name == "STL10":
        images = STL10(root=os.path.expanduser("Dataset/"), download=True, split="test",transform=preprocess)

    elif dataset_name == "SUN397":
        # images = SUN397(root=os.path.expanduser("Dataset/"), download=True,transform=preprocess)
        images = None
        
    elif dataset_name == "StanfordCars":
        images = StanfordCars(root=os.path.expanduser("Dataset/"), download=True, split="test",transform=preprocess)

    elif dataset_name == "UCF101":
        # images = UCF101(root=os.path.expanduser("Dataset/"),train=False,transform=preprocess)
        images = None
    
    else:
        images = None
        
    return images
#! FILE READ
# Open the prompts YAML
with open("data/prompts.yml", 'r') as stream:
    model_prompt = yaml.safe_load(stream)
# Open the models YAML
with open("data/models.yml", 'r') as stream:
    models = yaml.safe_load(stream)


output = {}
#! DATASET LOOP WITH EVALUATION
# datasets loop
for dataset_name in tqdm(model_prompt.keys()):
    results = []
    
    
    for model_item in models["Model Name"]:
        model, preprocess = clip.load(model_item, device=device)
  
        # load dataset
        images = dataset_loader(dataset_name)
        if images:
            loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
            zeroshot_weights = zeroshot_classifier(model_prompt[dataset_name]["classes"], model_prompt[dataset_name]["templates"])
            
            with torch.no_grad():
                top1, top5, n = 0., 0., 0.
                for i, (images, target) in enumerate(loader):
                    images = images.cuda()
                    target = target.cuda()
                    
                    # predict
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features @ zeroshot_weights

                    # measure accuracy
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                    top1 += acc1
                    top5 += acc5
                    n += images.size(0)

            top1 = (top1 / n) * 100
            result = str(round(top1,2))
            results.append(result)
            
        else:
            continue  
          
    output[dataset_name] = results  
    
    with open("test.txt", 'w') as f: 
        for key, value in output.items(): 
            f.write('%s:%s\n' % (key, value))
    
