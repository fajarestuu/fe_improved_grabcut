# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import argparse
import cv2
import os
from imutils import paths
import numpy as np

# construct the argument parser and parse the arguments


class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]



# load our serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join(["hed_model",
	"deploy.prototxt"])
modelPath = os.path.sep.join(["hed_model",
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)
d = 0
#loop each image in folder
for imgPath in paths.list_images("gambar"):
    

    # load the input image and grab its dimensions
    image = cv2.imread(imgPath)

    # resize image
    scale_percent = 10

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    dsize = (width, height)

    image = cv2.resize(image, dsize)
    (H, W) = image.shape[:2]

    # convert the image to grayscale, blur it, and perform Canny
    # edge detection
    print("[INFO] performing Canny edge detection...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)

    # construct a blob out of the input image for the Holistically-Nested
    # Edge Detector
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False)

    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    # export hed
    filename = "mask/mask_%d.jpg"%d
    cv2.imwrite(filename, hed)

    # rect
    rect = (50,50,W,H)

    # mask 1
    mask = np.zeros(image.shape[:2],np.uint8)
    newmask = cv2.imread("mask/mask_%d.jpg"%d, 0)

    # background and foreground 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #grabcut image 1
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = image*mask2[:,:,np.newaxis]
    # cv2.imshow("GRABCUT 1", img1)
    filename = "grabcut_no_mask/gb_%d.png"%d
    cv2.imwrite(filename, img1)

    # fixed mask
    mask[newmask <= 50] = cv2.GC_PR_FGD
    mask[newmask >= 205] = cv2.GC_BGD

    # grabcut image 2
    mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img1*mask[:,:,np.newaxis]
    # cv2.imshow("GRABCUT 2", img)
    filename = "grabcut_mask/gb_%d.jpg"%d
    cv2.imwrite(filename, img)
    
    # show the output edge detection results for Canny and
    # Holistically-Nested Edge Detection
    #cv2.imshow("Input", image)
    #cv2.imshow("HED", hed)
    #cv2.imshow("MASK", mask)
    #cv2.imshow("GRABCUT", img)
    d += 1
