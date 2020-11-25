import cv2
import numpy as np
import os
from utils_cpvton import *

def shape_from_contour(image, contour):
  dummy_mask = np.zeros((image.shape[0], image.shape[1], 3))
  dummy_mask = cv2.drawContours(
      dummy_mask, [contour], 0, (1, 0, 0), thickness=cv2.FILLED)
  x, y = np.where(dummy_mask[:, :, 0] == 1)
  inside_points = np.stack((x, y), axis=-1)
  return inside_points

def body_detection(image, parse):
  # binary thresholding by blue ?
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower_blue = np.array([0, 0, 120])
  upper_blue = np.array([180, 38, 255])
  mask = cv2.inRange(hsv, lower_blue, upper_blue)
  result = cv2.bitwise_and(image, image, mask=mask)

  # binary threshold by green ?
  b, g, r = cv2.split(result)
  filter = g.copy()
  _, mask = cv2.threshold(filter, 10, 255, 1)

  # at least original segmentation is FG
  mask[parse] = 1
  return mask

def neck_correction(path, image_file, fine_width=192, fine_height=256):
  image = cv2.imread(f'{path}images/{image_file}.jpg')
  parse_image = Image.open(f'output/parsing/val/{image_file}.png')
  gray = cv2.imread(f'output/parsing/val/{image_file}.png', cv2.IMREAD_GRAYSCALE)
  _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)           # if u take it from jpp, it is mono

  segmentation(seg_mask)
  image_mask = body_detection(image, seg_mask)
  plt.imshow(seg_mask)
  plt.show()
  upper_body = image_mask - seg_mask
  upper_body[upper_body > 0] = 20
  upper_body_vis = upper_body.copy()

  height, width = upper_body.shape
  upper_body[height//2:, :] = 0

  contours, hier = cv2.findContours(upper_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  if len(contours) > 0:
    cv2.drawContours(upper_body_vis, contours, -1, 255, 3)                      # draw in blue the contours that were founded
    c_neck = max(contours, key=cv2.contourArea)                                 # find the biggest area
    neck = shape_from_contour(image, c_neck)
    x, y, w, h = cv2.boundingRect(c_neck)
    cv2.rectangle(upper_body_vis, (x, y), (x + w, y + h), (170, 230, 0), 2)     # draw the book contour (in green)


  neck_mask = np.zeros((fine_height, fine_width)).astype(np.int)
  for each in neck:
      neck_mask[each[0]][each[1]] = 20

  parse_image = parse_image + neck_mask                                         # Add neck/skin to segmentation
  image_mask  = image_mask + seg_mask

  # handle overlapped pixels
  for i in range(1, 20):
      parse_image[parse_image == 20 + i] = i
  
  image_mask[seg_mask] = 1

  cv2.imwrite(f'{path}parse/{image_file}.png', parse_image)
  cv2.imwrite(f'{path}image-mask/{image_file}.png', image_mask)

  return {'image_mask': image_mask, 
          'parse_image': parse_image}

if __name__ == '__main__':
  for i in os.listdir('data/images'):
    name,ext = os.path.splitext(i)
    if ext in ['.jpg', '.png']:
      print(i)

      inputs = load_data(image_file=name)
  # output = neck_correction(inputs['image'], inputs['parse_image'])
  # visualize(output, 'Pre Processing', is_tensor=False)