import cv2
import imageio
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import fftpack as fpk
from tqdm import tqdm, trange
import time

# io.use_plugin('pil', 'imread')

# image = io.imread('tisato.jpg')

# print(image.shape)

# image_gray = rgb2gray(image)

# radon_img = np.zeros((image.shape[0] * 2, 180))

# center_x = image.shape[0] / 2 - 1
# center_y = image.shape[1] / 2 - 1

# for i in range(180):
#     proj = np.zeros((image.shape[0] * 2,))
#     proj_center = proj.shape[0] / 2 - 1
#     for x in range(image.shape[0]):
#         for y in range(image.shape[1]):
#             beta = i * np.pi / 180.
#             proj_x = np.floor((x - center_x) * np.cos(beta) + (y - center_y) * np.sin(beta) + proj_center)

#             proj_x = int(proj_x)
#             x = int(x)
#             y = int(y)
#             proj[proj_x] += image_gray[x, y]
#     radon_img[:, i] = proj

# plt.subplot(1, 2, 1)
# io.imshow(image_gray, 'matplotlib', cmap='gray')
# plt.subplot(1, 2, 2)
# io.imshow(radon_img, 'matplotlib', cmap='gray')
# plt.show()


def read_file(filename):
    m, n, k = [0, 0, 0]
    matrix = None
    with open(filename, 'r') as fobj:
        k, m, n = fobj.readline().strip().split(' ')[2: ]
        m = int(m)
        n = int(n)
        k = int(k)
        matrix = np.zeros((k, m, n))
        r, p, q = [0, 0, 0]
        for line in fobj.readlines():
            # print(line)
            if line == '\n':
                r += 1
                q = 0
                p = 0
            elif line == 'E':
                break
            else:
                data = line.split('\t')
                for datum in data:
                    matrix[r, p, q] = float(datum)
                    q += 1
                q = 0
                p += 1
            
    return matrix


def weighting(projection, r):
    weighted_projection = np.zeros_like(projection)
    x = np.zeros(projection.shape[1: 2])
    y = np.zeros(projection.shape[1: 2])
    x_min = -projection.shape[1] // 2 + 1
    x_max = projection.shape[1] // 2
    x[:] = np.linspace(x_min, x_max, projection.shape[1])
    y = np.flip(x.transpose())
    w = r / np.sqrt(r ** 2 + x ** 2 + y ** 2)
    print(w.max, w.min)
    weighted_projection = projection * w
    return weighted_projection


def backprojection(r, R, steps):
    '''
    R is a 3D matrix in the shape of [theta, channels, channels]
    '''
    channels = R.shape[1]
    stride = R.shape[0] // steps
    weighted_proj_volume = np.zeros((steps, channels, channels))
    reconstructed = np.zeros((channels, channels, channels))
    
    x = np.zeros(R.shape[1: 2])
    y = np.zeros(R.shape[1: 2])
    x_min = -R.shape[1] // 2 + 1
    x_max = R.shape[1] // 2
    x[:] = np.linspace(x_min, x_max, R.shape[1])
    y = np.flip(x.transpose())
    
    theta = np.arange(steps) * stride / 180.0 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=1), axis=2)
    x_volume = np.zeros_like(weighted_proj_volume)
    x_volume[:] = x
    y_volume = np.zeros_like(weighted_proj_volume)
    y_volume[:] = y
    x_volume = x_volume * np.cos(theta)
    y_volume = y_volume * np.sin(theta)
    epsilon = 1e-6
    w = r ** 2 / ((r + x_volume / 64.0 + y_volume / 64.0) ** 2 + epsilon ** 2)
    slices = (np.arange(steps) * stride).tolist()
    weighted_proj_volume = R[slices] * w
    
    bar = trange(R.shape[2])
    bar.set_description_str('Layer')
    for j in bar:
        proj_image_reranged = weighted_proj_volume[:, :, j]
        image = DiscreteIRadonTransform(proj_image_reranged, steps)
        bar.set_postfix({'Max': image.max(), 'Min': image.min()})
        reconstructed[j] = image
    return reconstructed


def DiscreteRadonTransform(image, steps):
    channels = image.shape[0]
    if channels % 2 == 0:
        pad_len = int(channels / 2)
    else:
        pad_len = int((channels - 1) / 2)
    image = np.pad(image, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    res = np.zeros((steps, channels + 2 * pad_len), dtype='float64')
    plt.ion()
    fig, axs = plt.subplots(1, 2)
    for s in range(steps):
        axs[0].cla()
        axs[1].cla()
        rotation = ndimage.rotate(
            image, -s * 180 / steps, reshape=False).astype('float64')
        axs[0].axis('off')
        axs[0].imshow(rotation, cmap='gray')
        # print(sum(rotation).shape)
        res[s] = sum(rotation) / channels
        axs[1].imshow(res, cmap='gray')
        plt.pause(0.03)
    plt.pause(2)
    plt.close(fig)
    plt.ioff()
    return res


def DiscreteIRadonTransform(R, steps):
    channels = R.shape[1]
    stride = R.shape[0] // steps
    image = np.zeros((channels, channels))
    # plt.ion()
    # fig, ax = plt.subplots()
    for s in range(steps):
        # ax.cla()
        degree = int(s * stride)
        bp = np.zeros_like(image)
        bp[:] = R[degree]
        rotation = ndimage.rotate(
            bp, degree, reshape=False).astype('float64')
        image += rotation
        # ax.axis('off')
        # ax.imshow(image, cmap='gray')
        # plt.pause(0.03)
    # plt.pause(0.5)
    # plt.close(fig)
    # plt.ioff()
    return image


# 读取原始图片
#image = cv2.imread("whiteLineModify.png", cv2.IMREAD_GRAYSCALE)
# image=imageio.imread('shepplogan.jpg').astype(np.float64)
#image = cv2.imread("whitePoint.png", cv2.IMREAD_GRAYSCALE)
def radon_2d():
    image = cv2.imread("tisato.jpg", cv2.IMREAD_GRAYSCALE)
    radon = DiscreteRadonTransform(image, 180)
    filtered_radon = np.zeros_like(radon)

    r_l_filter = np.abs(np.arange(-160, 160, 1, dtype=np.float64))
    # r_l_filter[: 80] = 0
    # r_l_filter[240: ] = 0
    r_l_filter = (r_l_filter - np.min(r_l_filter)) / \
        (np.max(r_l_filter) - np.min(r_l_filter))

    for i in range(radon.shape[0]):
        radon_w = fpk.fft(radon[i])
        radon_w_shifted = fpk.fftshift(radon_w)
        filtered_radon_w_shifted = radon_w_shifted * r_l_filter
        filtered_radon_w = fpk.fftshift(filtered_radon_w_shifted)
        filtered_radon_i = fpk.ifft(filtered_radon_w)
        filtered_radon[i] = np.real(filtered_radon_i).astype(np.float64)

    image1 = DiscreteIRadonTransform(radon, 180) / 180.
    # 裁剪有用图像片段
    image1 = image1[80: 240, 80: 240]

    image2 = DiscreteIRadonTransform(filtered_radon, 180)
    # 裁剪有用图像片段
    image2 = image2[80: 240, 80: 240]

    # 绘制原始图像和对应的sinogram图
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(image1, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(image2, cmap='gray')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.title('radon')
    plt.imshow(radon, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('filtered radon')
    plt.imshow(filtered_radon)
    plt.show()
    
def radon_3d():
    start = time.time()
    radon = read_file('./senbai/data666.txt')
    end = time.time()
    print('time cost: ', end - start)
    print(radon.shape)
    weighted_radon = weighting(radon, 100)
    print(weighted_radon.max(), weighted_radon.min())
    filtered_radon = np.zeros_like(weighted_radon, dtype=np.float64)
    length = weighted_radon.shape[1]
    r_l_filter = np.abs(np.arange(-length // 2, length // 2, 1, dtype=np.float64))
    # r_l_filter[: 80] = 0
    # r_l_filter[240: ] = 0
    r_l_filter = (r_l_filter - np.min(r_l_filter)) / \
        (np.max(r_l_filter) - np.min(r_l_filter))

    for i in range(weighted_radon.shape[0]):
        for j in range(weighted_radon.shape[2]):
            radon_w = fpk.fft(weighted_radon[i, :, j])
            radon_w_shifted = fpk.fftshift(radon_w)
            filtered_radon_w_shifted = radon_w_shifted * r_l_filter
            filtered_radon_w = fpk.fftshift(filtered_radon_w_shifted)
            filtered_radon_i = fpk.ifft(filtered_radon_w)
            filtered_radon[i, :, j] = np.real(filtered_radon_i).astype(np.float64)
    print(filtered_radon.shape)
    volume = backprojection(44.0, filtered_radon, 360)
    print(volume.shape)
    
    result_z = []
    for i in range(volume.shape[0]):
        result_z.append(volume[i])
        
    result_x = []
    for i in range(volume.shape[1]):
        result_x.append(volume[:, i, :])
        
    result_y = []
    for i in range(volume.shape[2]):
        result_y.append(volume[:, :, i])
    
    plt.ion()
    fig, ax = plt.subplots()
    # def update(frame):
    #     count, image = frame
    #     ax.axis('off')
    #     ax.set_title(f'Layer {count}')
    #     ax.imshow(image, cmap='gray')
    #     return ax
    # anime_z = animation.FuncAnimation(fig, update, frames=enumerate(result_z), interval=100, 
    #                                 repeat_delay=1000, save_count=len(result_z))
    # anime_x = animation.FuncAnimation(fig, update, frames=enumerate(result_x), interval=100, 
    #                                 repeat_delay=1000, save_count=len(result_x))
    # anime_y = animation.FuncAnimation(fig, update, frames=enumerate(result_y), interval=100, 
    #                                 repeat_delay=1000, save_count=len(result_y))
    for i in range(volume.shape[0]):
        ax.cla()
        ax.axis('off')
        ax.set_title(f'Layer{i}') 
        ax.imshow(volume[i], cmap='gray')
        max = volume[i].max()
        min = volume[i].min()
        print(f"max {max}, min {min}")
        plt.pause(0.1)
    plt.ioff()
    # anime_z.save('fdk_z.gif', writer='pillow')
    # anime_x.save('fdk_x.gif', writer='pillow')
    # anime_y.save('fdk_y.gif', writer='pillow')
    plt.show()
        
    
if __name__ == '__main__':
    # radon_2d()
    radon_3d()
