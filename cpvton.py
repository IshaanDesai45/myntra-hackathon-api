import torch
import os
import torch.nn.functional as F
from torch import nn
from networks import GMM, UnetGenerator, load_checkpoint
from dataloader import Dataloader
from utils_cpvton import *
from torchvision import transforms
 
class CPVTON:
  def __init__(self, gmm_path='checkpoints/GMM/gmm_final.pth', 
               tom_path='checkpoints/TOM/tom_final.pth', 
               use_cuda=torch.cuda.is_available()):
    self.gmm_path = gmm_path
    self.tom_path = tom_path
    self.use_cuda = use_cuda

  def predict(self, data, vis_data=False, vis_result=True):
    ToPIL = transforms.ToPILImage()
    inputs = Dataloader(data, stage='GMM')
    if vis_data:
      visualize(inputs, 'Data Preprocessing', 3, 4, (8, 10))

    result = self.GMM_Test(inputs)

    data['cloth']       = ToPIL(result['warped_cloth'].squeeze())
    data['cloth_mask']  = ToPIL(result['warped_mask'].squeeze())
    inputs = Dataloader(data, stage='TOM')

    result_tom = self.TOM_Test(inputs)

    result.update(result_tom)
    if vis_result:
      visualize(result, 'Tryon', figsize=(14, 10))
    return result['tryon']

  def GMM_Test(self, inputs):
    device = 'cuda' if self.use_cuda else 'cpu'

    model = GMM(use_cuda=self.use_cuda)
    load_checkpoint(model, self.gmm_path, use_cuda=self.use_cuda)
    model.to(device)
    model.eval()

    image      = inputs['image'].to(device)          # human image
    agnostic   = inputs['agnostic'].to(device)       # concat of shape, head, pose_map
    cloth      = inputs['cloth'].to(device)          # input cloth
    cloth_mask = inputs['cloth_mask'].to(device)     # cloth mask
    grid_image = inputs['grid_image'].to(device)     # grid image
    
    grid, theta = model(agnostic, cloth_mask)        # theta is not used

    warped_cloth = F.grid_sample(cloth,      grid, padding_mode='border')
    warped_mask  = F.grid_sample(cloth_mask, grid, padding_mode='zeros')
    warped_grid  = F.grid_sample(grid_image, grid, padding_mode='zeros')
    overlay = 0.7 * warped_cloth + 0.3 * image

    result = {
        'warped_cloth': warped_cloth,
        'warped_mask': warped_mask,
        'warped_grid': warped_grid,
        'overlay': overlay,
    }
    return {key:result[key].detach() for key in result}


  def TOM_Test(self, inputs):
    device = 'cuda' if self.use_cuda else 'cpu'

    model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(model, self.tom_path, use_cuda=self.use_cuda)
    model.to(device)
    model.eval()

    agnostic    = inputs['agnostic'].to(device)
    cloth       = inputs['cloth'].to(device)
    cloth_mask  = inputs['cloth_mask'].to(device)

    # outputs = model(torch.cat([agnostic, cloth], 1))  # CP-VTON
    outputs = model(torch.cat([agnostic, cloth, cloth_mask], 1))  # CP-VTON+
    rendered, composite = torch.split(outputs, 3, 1)
    rendered = torch.tanh(rendered)
    composite = torch.sigmoid(composite)
    tryon = cloth * composite + rendered * (1 - composite)

    result = {
        'tryon': tryon,
        'composite': composite,
        'rendered': rendered,
    }
    return {key:result[key].detach() for key in result}

if __name__ == '__main__':
  # data = load_files('', '010415', '004691')
  # for c in os.listdir('data/test/cloth'):
  data = load_files('')
  cpvton = CPVTON()
  cpvton.predict(data, vis_data=True)
  plt.show()