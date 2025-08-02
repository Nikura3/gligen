import os
import time

import torch
import numpy as np
import pandas as pd
import torchvision.utils
from PIL import Image
from diffusers import StableDiffusionGLIGENPipeline
import torchvision.transforms.functional as tf
import torchvision.utils
import math

from utils import logger

def NormalizeData(data,size=512):
    data=np.divide(data,size)
    return data


def readPromptsCSV(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict = {}
    for i in range(len(df)):
        entry = {'prompt': df.at[i, 'prompt']}
        # Dynamically find all obj/bbox columns and keep original naming
        for col in df.columns:
            if col.startswith('obj'):
                idx = col[3:]
                bbox_col = f'bbox{idx}'
                obj_val = df.at[i, col]
                bbox_val = df.at[i, bbox_col] if bbox_col in df.columns else None
                # Always include obj and bbox, even if NaN for retro compatibility
                entry[col] = obj_val
                if bbox_col in df.columns:
                    entry[bbox_col] = bbox_val
        conversion_dict[df.at[i, 'id']] = entry
    return conversion_dict


def main():

    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",safety_checker=None)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipe = pipe.to(device)

    height=512
    width=512
    seeds = range(1,9)

    #bench=make_tinyHRS()
    bench=readPromptsCSV(os.path.join("prompts","openSet.csv"))

    model_name="openSet-G"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
            
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(4))

    for id in ids:
        bboxes = []
        phrases = []
        # Dynamically find all obj/bbox columns for each id
        for col in bench[id]:
            if col.startswith('obj') and bench[id][col] is not None and not (isinstance(bench[id][col], (int, float)) and math.isnan(bench[id][col])):
                idx = col[3:]  # get the number after 'obj'
                bbox_col = f'bbox{idx}'
                if bbox_col in bench[id] and bench[id][bbox_col] is not None:
                    phrases.append(bench[id][col])
                    bboxes.append([int(x) for x in bench[id][bbox_col].split(',')])

        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",id)
        torch.cuda.empty_cache()

        gen_images=[]
        gen_bboxes_images=[]
        #BB: [xmin, ymin, xmax, ymax] normalized between 0 and 1
        normalized_boxes = NormalizeData(bboxes)

        start = time.time()
        for seed in seeds:
            print(f"Current seed is : {seed}")

            # start stopwatch
            start = time.time()

            if torch.cuda.is_available():
                g = torch.Generator('cuda').manual_seed(seed)
            else:
                g = torch.Generator('cpu').manual_seed(seed)

            images = pipe(prompt=bench[id]['prompt'],
                        gligen_phrases=phrases,
                        gligen_boxes=normalized_boxes,
                        gligen_scheduled_sampling_beta=1,
                        height=height,
                        width=width,
                        output_type="pil",
                        num_inference_steps=50,
                        generator=g,
                        negative_prompt='low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled').images

            # end stopwatch
            end = time.time()
            # save to logger
            l.log_time_run(start, end)

            #save the newly generated image
            image=images[0]
            image.save(output_path +"/"+ str(seed) + ".jpg")
            gen_images.append(tf.pil_to_tensor(image))

            #draw the bounding boxes
            image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),
                                                        torch.Tensor(bboxes),
                                                        labels=phrases,
                                                        colors=['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green'],
                                                        width=4,
                                                        font='font.ttf',
                                                        font_size=20)
            #list of tensors
            gen_bboxes_images.append(image)
            tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

        # save a grid of results across all seeds without bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + ".png")

        # save a grid of results across all seeds with bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + "_bboxes.png")
    
    # log gpu stats
    l.log_gpu_memory_instance()
    # save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process for ", model_name)


if __name__ == '__main__':
    main()
