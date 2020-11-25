from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import random

def visualize(input, title='', nrow=1, ncol=0, figsize=(10, 4), is_tensor=True):
  if ncol == 0:
    ncol = len(input)
  fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=figsize)
  fig.suptitle(title, fontsize=20)

  for i, n in enumerate(input):
    if is_tensor:
      image = input[n].cpu().numpy().copy().squeeze()
    else:
      image = input[n].copy().squeeze()
    if image.shape[0] == 3:
      image = np.transpose(image, (1, 2, 0))
    elif image.shape[0] in range(4, 23):
      image = np.sum(np.transpose(image, (1, 2, 0)), axis=2)

    if nrow > 1 and ncol > 1:
      ax[i//ncol][i%ncol].set_title(n)
      ax[i//ncol][i%ncol].imshow(image)
    else:
      ax[i].set_title(n)
      ax[i].imshow(image)
  
  fig.tight_layout()
  # fig.subplots_adjust(top=1)
  fig.show()

def load_data(path='', cloth_file='000010'):
  image       = Image.open(f'{path}data/images/image.jpg')
  cloth       = Image.open(f'{path}data/cloth/{cloth_file}_1.jpg')
  cloth_mask  = Image.open(f'{path}data/cloth-mask/{cloth_file}_1.jpg')
  grid        = Image.open(f'{path}grid.png')
  parse_image = Image.open(f'{path}output/parsing/val/{image_file}_0.png')

  with open(f'{path}output/pose/val/{image_file}_0.txt', 'r') as f:
    pose_label = [int(x) for x in f.readline().split(' ') if x != '']
    pose_label  = np.array(pose_label).reshape((-1, 2))
    pose_data = np.ones((pose_label.shape[0] + 2, pose_label.shape[1]+1))
    pose_data[:-2,:-1] = pose_label

    x = []
    for i in range(7):
      x.append([0, 0, 1])

    temp = [pose_data[0],  pose_data[7],  pose_data[12], pose_data[11], 
            pose_data[10], pose_data[13], pose_data[14], pose_data[15],
            pose_data[2],  pose_data[9],  x[0],  pose_data[3], *x[1:]]
    pose_data = np.array(temp)

    return {'image': image, 
          'cloth': cloth, 
          'cloth_mask': cloth_mask, 
          'grid': grid, 
          'image_mask': image_mask, 
          'parse_image': parse_image, 
          'pose_data': pose_data}

def load_files(path='', image_file='001337', cloth_file='001337'):
  image       = Image.open(f'{path}uploads/images/image.jpg')
  # cloth       = Image.open(f'{path}data/test/cloth/{cloth_file}_1.jpg')
  # cloth_mask  = Image.open(f'{path}data/test/cloth-mask/{cloth_file}_1.jpg')
  cloth       = Image.open(f'{path}data/test/cloth/{cloth_file}')
  cloth_mask  = Image.open(f'{path}data/test/cloth-mask/{cloth_file}')
  grid        = Image.open(f'{path}grid.png')

  image_mask  = Image.open(f'{path}output/parsing/image-mask.png')
  parse_image = Image.open(f'{path}output/parsing/image1._vis.png')

  with open(f'{path}data/test/pose/{image_file}_0_keypoints.json', 'r') as f:
    pose_label = json.load(f)
    pose_data  = pose_label['people'][0]['pose_keypoints']
    pose_data  = np.array(pose_data).reshape((-1, 3))

  return {'image': image, 
          'cloth': cloth, 
          'cloth_mask': cloth_mask, 
          'grid': grid, 
          'image_mask': image_mask, 
          'parse_image': parse_image, 
          'pose_data': pose_data}