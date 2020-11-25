import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from utils_cpvton import *

def Dataloader(data, stage='GMM'):
  fine_width, fine_height, r = 192, 256, 5
  transform = transforms.Compose([transforms.ToTensor(),                # converts numpy to torch.Tensor
                                  transforms.Normalize((0.5), (0.5))])  # Normalizes with mean=0.5, std=0.5
  # Cloth Transform
  # For GMM: cloth --> cloth        cloth_mask --> cloth_mask
  # For TOM: cloth --> warp-cloth   cloth_mask --> warp-mask  
  cloth = transform(data['cloth'])                                      # transform cloth image
  cloth_mask = (np.array(data['cloth_mask']) >= 128).astype(np.float32) # binary mask with value > 128
  cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)                # convert to torch and unsqueeze(h,w --> h,w,c)

  # Image Transform
  image = transform(data['image'])                                      # transform human image                                        
  parse_image = np.array(data['parse_image'])                           # create parse array
  parse_shape = (np.array(data['image_mask']) > 0).astype(np.float32)   # binary mask with value > 0

  # Parse Head
  parse_array_list = {'GMM': [4, 13], 'TOM': [2, 4, 9, 12, 13, 16, 17]} # parse head from its codes
  parse_head = (parse_image == 1).astype(np.float32)
  for i in parse_array_list[stage]:
    parse_head += (parse_image == i).astype(np.float32)

  # Parse Cloth Mask
  parse_cloth_mask = (parse_image == 5).astype(np.float32)
  for i in range(6, 8):
    parse_cloth_mask += (parse_image == i).astype(np.float32)

  # Parse Shape downsample
  parse_shape_ori = Image.fromarray((parse_shape*255).astype(np.uint8))
  parse_shape = parse_shape_ori.resize((fine_width//16, fine_height//16), Image.BILINEAR)
  parse_shape = parse_shape.resize((fine_width, fine_height), Image.BILINEAR)
  parse_shape = transform(parse_shape)

  # Keep an original copy of Parse Shape
  parse_shape_ori = parse_shape_ori.resize((fine_width, fine_height), Image.BILINEAR)
  parse_shape_ori = transform(parse_shape_ori)
  
  parse_head = torch.from_numpy(parse_head)
  parse_cloth_mask = torch.from_numpy(parse_cloth_mask).unsqueeze(0)

  # upper cloth
  parse_cloth = image * parse_cloth_mask + (1 - parse_cloth_mask)   # [-1,1], fill 1 for other parts
  parse_head = image * parse_head - (1 - parse_head)                # [-1,1], fill 0 for other parts

  # Pose Map
  pose_data = data['pose_data']
  point_num = pose_data.shape[0]
  pose_map = torch.zeros(point_num, fine_height, fine_width)
  pose_image = Image.new('L', (fine_width, fine_height))
  pose_draw = ImageDraw.Draw(pose_image)

  for i in range(point_num):
    one_map = Image.new('L', (fine_width, fine_height))
    draw = ImageDraw.Draw(one_map)
    pointx = pose_data[i, 0]
    pointy = pose_data[i, 1]
    if pointx > 1 and pointy > 1:
      draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
      pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
      # fnt = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 15)
      # pose_draw.text((pointx, pointy), str(i), font=fnt, fill=(255))
    one_map = transform(one_map)
    pose_map[i] = one_map[0]

  grid = transform(data['grid'])
  pose_image = transform(pose_image)
  parse_image = transform(parse_image)
  agnostic = torch.cat([parse_shape, parse_head, pose_map], 0)

  result = {
      'image': image,                           # human image
      'cloth': cloth,                           # input cloth
      'cloth_mask': cloth_mask,                 # cloth mask
      'parse_image': parse_image,               # human image masked
      'parse_cloth_mask': parse_cloth_mask,     # only cloth masked
      'parse_shape': parse_shape,               # human segmented mask (downsampled)
      'parse_shape_ori': parse_shape_ori,       # original body shape without blurring
      'parse_head': parse_head,                 # only head image
      'parse_cloth': parse_cloth,               # human current cloth only
      'pose_image': pose_image,                 # dots at pose coordinates
      'agnostic': agnostic,                     # concat of shape, head, pose_map
      'grid_image': grid,                       # grid image
  }

  # Very Important!!!, GMM takes only batch data, so batch axis is added
  return {key:result[key].unsqueeze_(0) for key in result}

if __name__ == '__main__':
  data = load_files(image_file=image_file, cloth_file=image_file, from_dataset=True)
  result = Dataloader(data, stage='GMM')
  visualize(result, '', 3, 4, (12, 12))