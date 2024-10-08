import numpy as np
import rasterio

def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)

def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform

def save_img(path,img,crs,transform):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()