
import argparse
import matplotlib.pyplot as plt

from ai import *

parser = argparse.ArgumentParser()
parser.add_argument('-imgpath', '--img_path', type=str, default='images/whisk.jpg')
parser.add_argument('-outname', '--save_prefix', type=str, default='saved')
opt = parser.parse_args()

# load colorizers
colorizer_siggraph17 = nn_out(pretrained=True).eval()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

plt.imsave('%s.jpg' % opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title('Input to the model')
plt.axis('off')


plt.subplot(2, 2, 3)
plt.imshow(out_img_siggraph17)
plt.title('Output (Colored Version)')
plt.axis('off')
plt.show()
