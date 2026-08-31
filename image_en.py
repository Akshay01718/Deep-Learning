import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

image=Image.open('/home/cs-ai-31/ak/image.png').convert('RGB')
transform=transforms.ToTensor()
img_tensor=transform(image)
gray=image.convert("1")
binary=gray.point(lambda p:255 if p>128 else 0)
brightness_factor=1.5
bright_image=img_tensor+brightness_factor
contrast_factor=1.3
mean=bright_image.mean(dim=(1,2),keepdim=True)
enhanced_image=(bright_image-mean)*contrast_factor+mean
enhanced_image=torch.clamp(enhanced_image,0.1)
to_pil=transforms.ToPILImage()
output=to_pil(enhanced_image)
gray_scale=transforms.Grayscale('/home/cs-ai-31/ak/image.png')
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original")
plt.axis("OFF")

plt.subplot(2,2,2)
plt.imshow(output)
plt.title("Enhanced")
plt.axis("OFF")

plt.subplot(2,2,3)
plt.imshow(binary,cmap="gray",interpolation="nearest")
plt.title("Binary")
plt.axis("OFF")
plt.subplot(2,2,4)
plt.imshow(gray)
plt.title("Grayscale")
plt.axis("OFF")
plt.show()
plt.subplot(2,2,2)
plt.imshow(output)
plt.title("Enhanced")
plt.axis("OFF")