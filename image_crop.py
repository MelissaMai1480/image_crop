# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 11:56:49 2018

@author: mai.mm.2
"""
        
from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
 
ImgPath = 'C:/Users/mai.mm.2/Desktop/may/' 
xmlPath = 'C:/Users/mai.mm.2/Desktop/xml/'
ProcessedPath = 'C:/Users/mai.mm.2/Desktop/1/'
 
imagelist = os.listdir(ImgPath)
for image in imagelist:
	image_pre, ext = os.path.splitext(image)
	imgfile = ImgPath + image 
	xmlfile = xmlPath + image_pre + '.xml'
	#storename,img = image_pre.split('_')
	DomTree = xml.dom.minidom.parse(xmlfile)
	annotation = DomTree.documentElement
 
	filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
	filename = filenamelist[0].childNodes[0].data
	objectlist = annotation.getElementsByTagName('object')
	
	i = 1
	for objects in objectlist:
		
		namelist = objects.getElementsByTagName('name')
		objectname = namelist[0].childNodes[0].data
 
		savepath = ProcessedPath + objectname
 
		if not os.path.exists(savepath):
			os.makedirs(savepath)
            
		#storepath = ProcessedPath + objectname + '/' + storename
        
		#if not os.path.exits(storepath):
		#	os .makedirs(storepath)
 
		bndbox = objects.getElementsByTagName('bndbox')
		cropboxes = []
 
		for box in bndbox:
			x1_list = box.getElementsByTagName('xmin')
			x1 = int(x1_list[0].childNodes[0].data)
			y1_list = box.getElementsByTagName('ymin')
			y1 = int(y1_list[0].childNodes[0].data)
			x2_list = box.getElementsByTagName('xmax')
			x2 = int(x2_list[0].childNodes[0].data)
			y2_list = box.getElementsByTagName('ymax')
			y2 = int(y2_list[0].childNodes[0].data)
 
			w = x2 - x1
			h = y2 - y1
 
			obj = np.array([x1,y1,x2,y2])
 
			cropboxes = np.tile(obj,(1,1))  
 
			img = Image.open(imgfile)
			for cropbox in cropboxes:
				cropedimg = img.crop(cropbox)
				cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.jpg')
				i += 1
print('all done')
                
    