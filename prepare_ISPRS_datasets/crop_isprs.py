import os
import numpy as np
from osgeo import gdal

# Color palette mapping
palette = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5       # Clutter/background
}

def convert_label(img):
    single_channel_label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for key, value in palette.items():
        mask = np.all(img == np.array(key), axis=-1)
        single_channel_label[mask] = value
    return single_channel_label

def process_image_gdal(img_path, is_label=False):
    ds = gdal.Open(img_path)
    img = ds.ReadAsArray()
    original_geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()


    if len(img.shape) == 2:  # Single-channel
        img = img[np.newaxis, ...]  # Add channel dimension
    img = np.nan_to_num(img)  # Replace NaN with 0

    if is_label and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # Convert to HxWxC
        img = convert_label(img)
        img = img[np.newaxis, ...]  # Add channel dimension back

    height, width = img.shape[1], img.shape[2]
    slice_size = 512

    # Calculate number of slices and step for overlap
    num_slices_height = (height - 1) // slice_size + 1
    num_slices_width = (width - 1) // slice_size + 1
    step_height = (height - slice_size) // (num_slices_height - 1) if num_slices_height > 1 else slice_size
    step_width = (width - slice_size) // (num_slices_width - 1) if num_slices_width > 1 else slice_size

    slices = []
    slice_geotransforms = []
    # Generate slices with overlap and calculate new geotransforms
    for i in range(num_slices_height):
        for j in range(num_slices_width):
            start_row = min(i * step_height, height - slice_size)
            start_col = min(j * step_width, width - slice_size)
            slice_img = img[:, start_row:start_row+slice_size, start_col:start_col+slice_size]

            # Calculate new geotransform for the slice
            new_geotransform = (
                original_geotransform[0] + start_col * original_geotransform[1],
                original_geotransform[1],
                0.0,
                original_geotransform[3] + start_row * original_geotransform[5],
                0.0,
                original_geotransform[5]
            )
            slices.append(slice_img)
            slice_geotransforms.append(new_geotransform)

    return slices, slice_geotransforms, projection

# Directories
dirs = ['top', 'dsm', 'label']
new_dirs = ['opt', 'DSM', 'gt_ss_mask']

# Create new directories for slices
for new_dir in new_dirs:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# Iterate through each directory and process images
for dir_idx, dir_name in enumerate(dirs):
    for filename in os.listdir(dir_name):
        img_path = os.path.join(dir_name, filename)
        slices, slice_geotransforms, projection = process_image_gdal(img_path, is_label=(dir_name == 'label'))

        # Save slices
        for idx, (slice_img, geotransform) in enumerate(zip(slices, slice_geotransforms)):
            slice_filename = f"{filename.split('.')[0]}_slice_{idx}.tif"
            driver = gdal.GetDriverByName('GTiff')
            data_type = gdal.GDT_Float32 if dir_name == 'dsm' else gdal.GDT_Byte
            channels = slice_img.shape[0]

            out_ds = driver.Create(os.path.join(new_dirs[dir_idx], slice_filename), 512, 512, channels, data_type)
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)

            for ch in range(channels):
                out_band = out_ds.GetRasterBand(ch + 1)
                out_band.WriteArray(slice_img[ch, :, :])
                out_band.FlushCache()

            out_ds = None  # Close the dataset

print("Processing complete.")