# Style-GAN-Prior
This project is an extension to the article [Style Generator Inversion for Image Enhancement and Animation](http://www.vision.huji.ac.il/style-image-prior) by Aviv Gabbay and Yedid Hoshen [(GitHub)](https://github.com/avivga/style-image-prior).
Two additional applications are introduced to this method: Image deblurring using a known blur kernel and image recoloring of gray images to RGB images.


## Deblurring
In this task, images are deblurred using the style-GAN-prior and a known blur kernel.

Input image:

![image](https://user-images.githubusercontent.com/61732335/86510827-0adec000-bdfc-11ea-9728-e712767dccea.png)

| Gaussian Blur | Uniform Blur | Motion Blur |
| :---: | :---: | :---: |
| ![image](https://user-images.githubusercontent.com/61732335/86510828-103c0a80-bdfc-11ea-909f-5ed122cfe1c0.png) | ![image](https://user-images.githubusercontent.com/61732335/86510875-87719e80-bdfc-11ea-8021-baa384f88a9c.png) | ![image](https://user-images.githubusercontent.com/61732335/86510958-331aee80-bdfd-11ea-8847-204735f3fc1f.png) |
| Gaussian Reconstruction | Uniform Reconstruction | Motion Reconstruction |
| ![image](https://user-images.githubusercontent.com/61732335/86510829-129e6480-bdfc-11ea-995c-d86325b74c27.png) | ![image](https://user-images.githubusercontent.com/61732335/86510878-8a6c8f00-bdfc-11ea-8478-08bbae7cada4.png) | ![image](https://user-images.githubusercontent.com/61732335/86510959-357d4880-bdfd-11ea-932d-7ec980d0319c.png) |

Input image:

![image](https://user-images.githubusercontent.com/61732335/86511019-e2f05c00-bdfd-11ea-8d00-b2c4164041fc.png)

| Gaussian Blur | Uniform Blur | Motion Blur |
| :---: | :---: | :---: |
| ![image](https://user-images.githubusercontent.com/61732335/86511021-e552b600-bdfd-11ea-9b4e-870f27bc14ad.png) | ![image](https://user-images.githubusercontent.com/61732335/86511060-1b903580-bdfe-11ea-91b5-43259dbd9812.png) | ![image](https://user-images.githubusercontent.com/61732335/86511075-45e1f300-bdfe-11ea-9d2f-9670adf9268c.png) |
| Gaussian Reconstruction | Uniform Reconstruction | Motion Reconstruction |
| ![image](https://user-images.githubusercontent.com/61732335/86511023-e7b51000-bdfd-11ea-9b75-8d2a964fcec0.png) | ![image](https://user-images.githubusercontent.com/61732335/86511062-1e8b2600-bdfe-11ea-9273-d4350cf234bf.png) | ![image](https://user-images.githubusercontent.com/61732335/86511076-4a0e1080-bdfe-11ea-91cd-f6ea0bc84f45.png) |


## Recoloring
In this task, gray images are recolored using the style-GAN-prior.
The conversion of an RGB image to a gray image was performed using the formula: Y = 0.2126R + 0.7152G + 0.0722B

| RGB Image | Gray Image | Recolored Image |
| :---: | :---: | :---: |
| ![image](https://user-images.githubusercontent.com/61732335/86511220-a291dd80-bdff-11ea-8c1c-c1889c3ff102.png) | ![image](https://user-images.githubusercontent.com/61732335/86511223-a4f43780-bdff-11ea-81e5-04a51666b56f.png) | ![image](https://user-images.githubusercontent.com/61732335/86511225-a6bdfb00-bdff-11ea-912f-0b87ec3b1540.png) |
| ![image](https://user-images.githubusercontent.com/61732335/86511226-a9b8eb80-bdff-11ea-8a71-9a610da6226f.png) | ![image](https://user-images.githubusercontent.com/61732335/86511227-ac1b4580-bdff-11ea-8d58-a20948402e50.png) | ![image](https://user-images.githubusercontent.com/61732335/86511228-ae7d9f80-bdff-11ea-8f63-3eafc435c1bf.png) |

## Usage
### Getting started
1. Clone the official [StyleGAN](https://github.com/NVlabs/stylegan) repository. 
2. Add the local StyleGAN project to PYTHONPATH.

### Style Image Prior for Deblurring
Deblurring blurry images using a known blur kernel:
```
deblurring.py --blurred-imgs-dir <blurred-input-dir> --deblurred-imgs-dir <deblurred-output-dir>
    --latents-dir <output-latents-dir> --blur-kernel <blur-method>
    [--original-imgs-dir <originl-imgs-dir>]
    [--blurred-img-size BLURRED_IMG_HEIGHT BLURRED_IMG_WIDTH]
    [--learning-rate LEARNING_RATE]
    [--total-iterations TOTAL_ITERATIONS]
```
**Notes:**
1. The optional values for the blur-kernel flag are: gaussian/uniform/motion-vertical/motion-horizontal.
2. If the original-imgs-dir is specified, the chosen blur-kernel will be first applied to all images in this dir and the blurred images will be saved in the blurred-imgs-dir.
3. The default input image size is 256X256X3. This can be changed using the --blurred-img-size flag.

### Style Image Prior for Image Recoloring:
Given a gray face image, reconstruct an RGB image:
```
coloring.py --gray-imgs-dir <gray-input-dir> --colored-imgs-dir <colored-output-dir>
    --latents-dir <output-latents-dir>
    [--original-imgs-dir <originl-imgs-dir>]
    [--img-size GRAY_IMG_HEIGHT GRAY_IMG_WIDTH]
    [--learning-rate LEARNING_RATE]
    [--total-iterations TOTAL_ITERATIONS]
```
**Notes:**
1. If the original-imgs-dir is specified, all images in this dir will be first converted to gray images using the formula Y = 0.2126R + 0.7152G + 0.0722B and saved in the gray-imgs-dir.
2. The default input image size is 256X256X3. This can be changed using the --img-size flag.

