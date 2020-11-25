from cpvton import CPVTON
from preprocessing import *
from torchvision import transforms
from utils_cpvton import *
import numpy as np
from PIL import Image
import cv2 as cv
import torch

class Model(object):
  def __init__(self, gmm_path='checkpoints/GMM/gmm_final.pth', 
               tom_path='checkpoints/TOM/tom_final.pth', 
               use_cuda=torch.cuda.is_available()):
    self.cpvton = CPVTON(gmm_path, tom_path, use_cuda=use_cuda)
    self.transform = transforms.Compose([transforms.Normalize((-1), (2)),
                                         transforms.ToPILImage()])

  def predict(self, data, keep_back=True, need_dilate=True):
    # result = neck_correction(data['image'], data['parse_image'])
    # data.update(result)

    out_image = self.cpvton.predict(data, vis_data=False, vis_result=False)
    out_image = self.transform(out_image.squeeze())

    # plt.imshow(out_image)
    # plt.show()

    if keep_back:
      parse = np.array(data['parse_image'])
      if len(parse.shape) == 2:
        parse = parse.reshape((256, 192, 1))

      cloth_mask = np.array(parse == 5, dtype='float32')
      if need_dilate:
        cloth = cloth_mask[:, :, 0]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
        dilated = cv.dilate(cloth, kernel)
        dilated = cv.blur(dilated, (14, 14))

        new_cloth = Image.fromarray((dilated*255))
        new_cloth = new_cloth.resize((192//10, 256//10), Image.BILINEAR)
        new_cloth = new_cloth.resize((192, 256), Image.BILINEAR)

        new_cm = np.array(new_cloth)
        new_cm = np.array(new_cm/255, dtype='float32')
        cloth_mask = np.resize(new_cm, (256, 192, 1))
      out_image = data['image']*(1-cloth_mask)+out_image*cloth_mask

    out_image = cv.cvtColor(out_image, cv.COLOR_BGR2RGB)
    return out_image

if __name__ == '__main__':
  model = Model()
  image_file='011203'
  cloth_file='000274_1.jpg'
  data = load_files('', image_file, cloth_file)
  output = model.predict(data)
  cv.imwrite(f'output/result/{image_file}.jpg', output)