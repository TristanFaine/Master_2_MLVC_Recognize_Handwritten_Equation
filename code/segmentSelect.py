################################################################
# segmentSelect.py
#
# Program that select hypothesis of segmentation
#
# Author: H. Mouchere, Dec. 2018
# Copyright (c) 2018, Harold Mouchere
################################################################
import sys
import random
import itertools
import sys, getopt
import torch
import numpy as np
from convertInkmlToImg import parse_inkml,get_traces_data, getStrokesFromLG, convert_to_imgs, parseLG
from skimage.io import imsave
from torchvision.transforms import Compose, ToTensor, Normalize
from modules import SegmentSelector, AlexNet

from globals import *

#TODO: refer to a trained model that we'll put in a data/ or models/ folder.
model = AlexNet()
model.load_state_dict(dict(torch.load('segmentSelector.nn')))
img_to_tensor = Compose([
  ToTensor(),
  Normalize((0.5,), (0.5,))])

def usage():
    print ("usage: python3 [-o fname] [-s] segmentSelect.py inkmlfile lgFile ")
    print ("     inkmlfile  : input inkml file name ")
    print ("     lgFile     : input LG file name")
    print ("     -o fname / --output fname : output file name (LG file)")
    print ("     -s         : save hyp images")


"""
take an hypothesis (from LG = list of stroke index), select the corresponding strokes (from allTraces) and 
return the probability of being a good segmentation [0:1]  
"""
def computeProbSeg(alltraces, hyp, saveIm = False):
    im = np.squeeze(np.asarray(convert_to_imgs(get_traces_data(alltraces, hyp[1]), IMG_SIZE)))
    if saveIm:
        imsave(hyp[0] + '.png', im)
    im_tensor = img_to_tensor(im)
    # Give it a "batch size" of 1
    im_tensor = im_tensor.unsqueeze(0)

    # Use softmax to set probabilities between 0 and 1
    output = torch.nn.functional.softmax(model(im_tensor),dim=1)
    return output[0][0].item()

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "so:", ["output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) < 2:
        print("Not enough parameters")
        usage()
        sys.exit(2)
    inputInkml = args[0]
    inputLG = args[1]
    saveimg = False
    outputLG = ""
    for o, a in opts:
        if o in ("-s"):
            saveimg = True
        elif o in ("-o", "--output"):
            outputLG = a
        else:
            usage()
            assert False, "unhandled option"
    traces = parse_inkml(inputInkml)
    hyplist = open(inputLG, 'r').readlines()
    hyplist = parseLG(hyplist)
    output = ""
    for h in hyplist:
        prob = computeProbSeg(traces, h, saveimg)
        #### select your threshold
        #TODO: put threshold here.
        if prob > 0.5: 
          output += "O,"+ h[0]+",*,"+str(prob)+","+",".join([str(s) for s in h[1]]) + "\n"
    if outputLG != "":
        with open(outputLG, "w") as text_file:
            print(output, file=text_file)
    else:
        print(output)


if __name__ == "__main__":
    # execute only if run as a script
    main()
