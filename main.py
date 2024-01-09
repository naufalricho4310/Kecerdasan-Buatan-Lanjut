import torch
import numpy as np
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    result = model(frame)
    
    cv2.imshow('frame', np.squeeze(result.render()))
    toggle = cv2.waitKey(2)
    if toggle == ord("q"):
    	break
 
 
 
 
 # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp47/wight/last.pt', )

# img = os.path.join('datasets', 'images', 'mouse1.jpg')
# results = model(img)

# results.print()
# for label in labels:
#     print('collecting images for {}'.format(label))
#     time.sleep(4)
    
#     for img_num in range (number_imgs):
#         print('collecting image for {}, image number {}'.format(label, img_num))
        
#         ret, frame = cap.read()
        
#         imgName = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
#         cv2.imwrite(imgName, frame)
        
#         cv2.imshow('image collection', frame)
#         time.sleep(2)
#         toggle = cv2.waitKey(2)
#         if toggle == ord("q"):
#     		    break
