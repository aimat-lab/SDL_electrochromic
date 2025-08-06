import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.filters import sobel
from PIL import Image, ImageFilter
from typing import Tuple  


# Function to apply gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    lut = [pow(i / 255.0, inv_gamma) * 255 for i in range(256)]
    return image.point(lut)


# Define a function to apply a 1D vertical blur to each column
def vertical_blur(image, radius=5, gamma=.2):
    # Apply gamma correction (lower gamma for higher contrast)
    image = adjust_gamma(image, gamma=gamma)  # Try gamma < 1.0 to boost contrast in darker areas
    image_array = np.array(image, dtype=np.float32)
    _, width = image_array.shape
    # Pad the image with reflection at the top and bottom to avoid white bands
    padded_image_array = np.pad(image_array, ((radius, radius), (0, 0)), mode='reflect')
    blurred_image = np.zeros_like(padded_image_array)
    # Apply vertical blur (box blur) to each column
    for col in range(width):
        column = padded_image_array[:, col]  # Extract the column
        blurred_column = np.convolve(column, np.ones((radius,)) / radius, mode='same')
        blurred_image[:, col] = blurred_column  # Place the blurred column back
    # Remove the padding to return to the original image size
    blurred_image = blurred_image[radius:-radius, :]
    # Subtract the local mean from the original image
    result_np = image_array - blurred_image
    # Rescale the result to the range [0, 255]
    min_val = result_np.min()
    max_val = result_np.max()
    if max_val - min_val > 0:  # Avoid division by zero
        result_np = (result_np - min_val) / (max_val - min_val) * 255
    else:
        result_np = np.zeros_like(result_np)  # If all values are the same, set the result to zero
    # Convert to uint8
    result_np = result_np.astype(np.uint8)
    # Convert the result back to a PIL image
    result_image = Image.fromarray(result_np)
    return result_image


# Define a function to apply a 1D vertical blur to each column
def horizontal_blur(image, radius=5, gamma=.2):
    # Apply gamma correction (lower gamma for higher contrast)
    image = adjust_gamma(image, gamma=gamma)  # Try gamma < 1.0 to boost contrast in darker areas
    image_array = np.array(image, dtype=np.float32)
    height, _ = image_array.shape
    # Pad the image with reflection at the sides to avoid white bands
    padded_image_array = np.pad(image_array, ((0, 0), (radius, radius)), mode='reflect')
    blurred_image = np.zeros_like(padded_image_array)
    # Apply horizontal blur (box blur) to each column
    for row in range(height):
        line = padded_image_array[row, :]  # Extract the row
        blurred_row = np.convolve(line, np.ones((radius,)) / radius, mode='same')
        blurred_image[row, :] = blurred_row  # Place the blurred row back
    # Remove the padding to return to the original image size
    blurred_image = blurred_image[:, radius:-radius]
    # Subtract the local mean from the original image
    result_np = image_array - blurred_image
    # Rescale the result to the range [0, 255]
    min_val = result_np.min()
    max_val = result_np.max()
    if max_val - min_val > 0:  # Avoid division by zero
        result_np = (result_np - min_val) / (max_val - min_val) * 255
    else:
        result_np = np.zeros_like(result_np)  # If all values are the same, set the result to zero
    # Convert to uint8
    result_np = result_np.astype(np.uint8)
    # Convert the result back to a PIL image
    result_image = Image.fromarray(result_np)
    return result_image


def normalize_brightness(image, target_mean=128):
    # Convert the image to a NumPy array
    image_np = np.array(image, dtype=np.float32)
    # Calculate the current mean brightness of the image
    current_mean = np.mean(image_np)
    # Compute the scaling factor to adjust the mean brightness to the target
    scaling_factor = target_mean / current_mean if current_mean != 0 else 1
    # Scale the pixel values
    normalized_image_np = image_np * scaling_factor
    # Clip values to ensure they are within the valid range [0, 255]
    normalized_image_np = np.clip(normalized_image_np, 0, 255).astype(np.uint8)
    # Convert the result back to a PIL image
    normalized_image = Image.fromarray(normalized_image_np)
    return normalized_image


def subtract_local_mean(image, radius=10, gamma=.2):
    # Apply gamma correction (lower gamma for higher contrast)
    image = adjust_gamma(image, gamma=gamma)  # Try gamma < 1.0 to boost contrast in darker areas
    # plt.imshow(image, cmap='gray')
    # plt.show()
    # Apply a mean filter using the ImageFilter.BoxBlur
    local_mean = image.filter(ImageFilter.BoxBlur(radius))
    # Convert images to NumPy arrays for manipulation
    original_np = np.array(image, dtype=np.float32)
    local_mean_np = np.array(local_mean, dtype=np.float32)
    # Subtract the local mean from the original image
    result_np = original_np - local_mean_np
    # Rescale the result to the range [0, 255]
    min_val = result_np.min()
    max_val = result_np.max()
    if max_val - min_val > 0:  # Avoid division by zero
        result_np = (result_np - min_val) / (max_val - min_val) * 255
    else:
        result_np = np.zeros_like(result_np)  # If all values are the same, set the result to zero
    # Convert to uint8
    result_np = result_np.astype(np.uint8)
    # Convert the result back to a PIL image
    result_image = Image.fromarray(result_np)
    return result_image


def binarize(img, threshold, invert=False):
    img2 = img
    # Threshold
    if not invert:
        img2 = img2.point( lambda p: 255 if p > threshold else 0 )
    else:
        img2 = img2.point( lambda p: 0 if p > threshold else 255 )
    # To mono
    img2 = img2.convert('1')
    return(img2)


def scale_back(img: np.ndarray):
    return ((img - img.min())*255/(img.max()-img.min())).astype(int)


def bounding_box(
          img_size: Tuple[int, int], 
          pad: Tuple[float, float] = (.17, .17),
          ):
    w, h = img_size
    xmin, xmax = int(w*pad[0]), w - int(w*pad[0])
    ymin, ymax = int(h*pad[1]), h - int(h*pad[1])
    # ymin, ymax = int(h*pad[1]), h - int(h*.4)
    return (xmin, ymin, xmax, ymax)
    

def get_image(
          path: str, 
          pad: Tuple[float, float] = (.10, .10), 
          newsize: Tuple[int,int] = (600, 600),
          grey_scale = True,
          blur=True,
          ):
    img = Image.open(path) 
    bbox = bounding_box(img.size, pad)
    img = img.crop(bbox).resize(newsize)
    if grey_scale:
        img = img.convert('L')
    return img


def get_hist(img: Image):
    img_arr = np.array(img)
    img_arr = abs(img_arr - img_arr.mean())
    counts, bins = np.histogram(img_arr, bins=100)
    return counts, bins, img_arr.sum()


def filter_img(img):
    fltrd_img = sobel(np.array(img))
    return fltrd_img


def plot_sorted(sort, type='bright', name=''):
    fig, axsg = plt.subplots(5, 7, figsize=(30,30))
    axs = axsg.flatten()
    for i, n in enumerate(sort):
        img = get_image('./'+type+'/SD169-'+str(n)+'.bmp', grey_scale=False)
        axs[i].imshow(img)
        axs[i].set_title('SD169-'+str(n)+'.bmp')
    plt.tight_layout()
    plt.savefig(type+'_sorted_'+name+'.png')
    plt.close()


def analyse_images(folder):
    nums_spots_bright_all = []
    size_largest_bright_all = []
    num_spots_above_size_threshold_bright_all = []
    nums_spots_dark_all = []
    size_largest_dark_all = []
    num_spots_above_size_threshold_dark_all = []
    num_bins_above_threshold_dark_all = []
    num_bins_above_threshold_bright_all = []

    ratio_bright_all = []
    ratio_dark_all = []

    LOW_THRESH = 4
    def small_blob_condition(x):
        low_threshold = LOW_THRESH
        high_threshold = 400
        if isinstance(x, int):
            return x>low_threshold and x<high_threshold
        else:
            return [xi>low_threshold and xi<high_threshold for xi in x]
        
    thresholds_bright = [180, 175, 150] 
    thresholds_dark = [80, 100, 120] 
    np.savetxt("analysis/images/thresholds_bright.txt", thresholds_bright)
    np.savetxt("analysis/images/thresholds_dark.txt", thresholds_dark)

    filenames = {}
    list_todo = []
    for fn in os.listdir("data/images"):
        if ".bmp" in fn:
            fn = fn.replace(".bmp", "")
            ID = fn.split("_")[0]
            if ID not in filenames.keys():
                list_todo.append(ID)
                filenames[ID] = {}
            if "dark" in fn:
                filenames[ID]["dark"] = fn
            elif "bright" in fn:
                filenames[ID]["bright"] = fn
            else:
                exit("ERROR in filename %s"%(fn))
    list_todo = sorted(list_todo)
        
    for i, ID in enumerate(list_todo):
        filename1 = filenames[ID]["bright"]
        filename2 = filenames[ID]["dark"]
        print("load images %s %s.bmp %s.bmp"%(folder, filename1, filename2))
        
        num_add_panels = max([len(thresholds_bright), len(thresholds_dark)])
        fig, axes = plt.subplots(2, 2+num_add_panels, figsize = (10+5*num_add_panels, 10))
        
        if not os.path.exists("./%s/%s.bmp"%(folder, filename1)) or not os.path.exists("./%s/%s.bmp"%(folder, filename2)):
            print("skip %s, files not found"%(filename1))
            continue
            
        img_bright = get_image("./%s/%s.bmp"%(folder, filename1), pad=(.23, .23)) # 29, 29
        img_bright = normalize_brightness(img_bright, target_mean=250)
        img_bright = vertical_blur(img_bright, radius=100, gamma=.27)
        # img_bright = horizontal_blur(img_bright, radius=100, gamma=.27)
        img_bright = subtract_local_mean(img_bright, radius=10, gamma=1.)
        img_bright = normalize_brightness(img_bright, target_mean=230)
        
        img_dark = get_image("./%s/%s.bmp"%(folder, filename2), pad=(.23, .23))
        img_dark = subtract_local_mean(img_dark, radius=10, gamma=2.5)
        img_dark = normalize_brightness(img_dark, target_mean=30)
        bins = np.linspace(-1e-9, 255+1e-9, 100)
        
        # plot the images 
        axes[0][0].imshow(np.array(img_bright), cmap='gray', interpolation = None, vmin=0, vmax=255)
        axes[1][0].imshow(np.array(img_dark), cmap='gray', interpolation = None, vmin=0, vmax=255)
        
        # plot the histograms
        counts_bright, bins_bright, bars_bright = axes[0][1].hist(np.array(img_bright).flatten(), bins=bins)
        axes[0][1].set_yscale('log', nonpositive='clip')
        axes[0][1].set_xlim([50, 255])
        counts_dark, bins_dark, bars_dark = axes[1][1].hist(np.array(img_dark).flatten(), bins=bins)
        axes[1][1].set_yscale('log', nonpositive='clip')
        axes[1][1].set_xlim([0, 255])
        for t_idx, t in enumerate(thresholds_bright[::-1]):
            axes[0][1].plot([t, t], [1e-1, np.max(counts_bright)*1.1], "k-")
            axes[0][1].text(t+2, np.max(counts_bright)*0.6*(0.6)**t_idx, "%i"%t)
        for t_idx, t in enumerate(thresholds_dark):
            axes[1][1].plot([t, t], [1e-1, np.max(counts_dark)*1.1], "k-")
            axes[1][1].text(t+2, np.max(counts_dark)*0.6*(0.6)**t_idx, "%i"%t)
        axes[0][1].set_ylim([5e-1, np.max(counts_bright)*1.1])
        axes[1][1].set_ylim([5e-1, np.max(counts_dark)*1.1])
            
        num_bins_above_threshold_bright = len(np.where(counts_bright>=100)[0])
        num_bins_above_threshold_bright_all.append(num_bins_above_threshold_bright)
        num_bins_above_threshold_dark = len(np.where(counts_dark>=100)[0])
        num_bins_above_threshold_dark_all.append(num_bins_above_threshold_dark)
        
        # BRIGHT
        white_ratio_bright = []
        nums_spots_bright = []
        sizes_spots_bright = []
        size_largest_bright = []
        num_spots_above_size_threshold_bright = []
        size_threshold = 36 # in pixel
        for threshold_idx, threshold_bright in enumerate(thresholds_bright):
            img_bright_binarized = binarize(img_bright, threshold_bright, invert=True)
            # np.asarray(img_bright_binarized).sum()
            # convert to cv2
            img_bright_binarized_cv = np.array(img_bright_binarized).astype(np.int8).copy()
            # identify the spots and count them
            nb_blobs_bright, im_with_separated_blobs_bright, stats_bright, _ = cv2.connectedComponentsWithStats(img_bright_binarized_cv)

            widths = []
            hights = []
            sizes_here = []
            for i in range(nb_blobs_bright):
                coordinates = np.where(im_with_separated_blobs_bright == i)
                if len(coordinates[0])>0:
                    c_here = img_bright_binarized_cv[coordinates[0][0]][coordinates[1][0]]                
                    if c_here==1:
                        sizes_here.append(len(coordinates[0]))
                        w = np.max(coordinates[0]) - np.min(coordinates[0])
                        h = np.max(coordinates[1]) - np.min(coordinates[1])
                        widths.append(w)
                        hights.append(h)
            widths = np.array(widths)
            hights = np.array(hights)
            lenghts = np.max(np.vstack([widths, hights]), axis = 0)
            
            # compute ratio
            img_dims = np.array(img_bright).shape
            white_ratio_bright.append(100*sum([size for size in sizes_here if small_blob_condition(size)])/(img_dims[0]*img_dims[1]))

            nums_spots_bright.append(nb_blobs_bright-1)
            sizes_here = np.array(sorted(sizes_here))
            sizes_spots_bright.append(sizes_here)
            # nb_blobs_bright_small = len(np.where(sizes_here<400)[0])
            nb_blobs_bright_small = len(np.where(small_blob_condition(sizes_here))[0])
            nb_blobs_bright_large = len(np.where(sizes_here<1000000)[0]) - nb_blobs_bright_small - len(np.where(sizes_here<=LOW_THRESH)[0])
            nb_blobs_bright_long = len(np.where(lenghts>70)[0])
            if len(sizes_here)>0:
                size_largest_bright.append(sizes_here[-1])
                num_spots_above_size_threshold_bright.append(len(np.where(np.array(sizes_here)>size_threshold)[0]))
            else:
                size_largest_bright.append(0)
                num_spots_above_size_threshold_bright.append(0)
            
            axes[0][2+threshold_idx].imshow(img_bright_binarized, cmap='gray', interpolation = None, vmin=0, vmax=255)
            axes[0][2+threshold_idx].text(10, 36, "%i small spots"%(nb_blobs_bright_small), fontsize = 12, c="lightgreen")
            axes[0][2+threshold_idx].text(10, 36+36, "%i large spots (%s)"%(nb_blobs_bright_large, np.sort(sizes_here)[-3:]), fontsize = 12, c="lightgreen")
            axes[0][2+threshold_idx].text(10, 36+2*36, "%i long spots (%s)"%(nb_blobs_bright_long, np.sort(lenghts)[-3:]), fontsize = 12, c="lightgreen")
            axes[0][2+threshold_idx].text(10, 36+3*36, "ratio (%s)"%(white_ratio_bright[-1]), fontsize = 12, c="lightgreen")
            
        # DARK
        white_ratio_dark = []
        nums_spots_dark = []
        sizes_spots_dark = []
        size_largest_dark = []
        num_spots_above_size_threshold_dark = []
        size_threshold = 36 # in pixel
        for threshold_idx, threshold_dark in enumerate(thresholds_dark):
            # binarize
            img_dark_binarized = binarize(img_dark, threshold_dark, invert=False)
            # convert to cv2
            img_dark_binarized_cv = np.array(img_dark_binarized).astype(np.int8).copy()
            # identify the spots and count them
            nb_blobs_dark, im_with_separated_blobs_dark, stats_dark, _ = cv2.connectedComponentsWithStats(img_dark_binarized_cv)
            widths = []
            hights = []
            sizes_here = []
            for i in range(nb_blobs_dark):
                coordinates = np.where(im_with_separated_blobs_dark == i)
                if len(coordinates[0])>0:
                    c_here = img_dark_binarized_cv[coordinates[0][0]][coordinates[1][0]]                
                    if c_here==1:
                        sizes_here.append(len(coordinates[0]))
                        w = np.max(coordinates[0]) - np.min(coordinates[0])
                        h = np.max(coordinates[1]) - np.min(coordinates[1])
                        widths.append(w)
                        hights.append(h)
            widths = np.array(widths)
            hights = np.array(hights)
            lenghts = np.max(np.vstack([widths, hights]), axis = 0)

            # compute ratio
            img_dims = np.array(img_bright).shape
            white_ratio_dark.append(100*sum([size for size in sizes_here if small_blob_condition(size)])/(img_dims[0]*img_dims[1]))

            nums_spots_dark.append(nb_blobs_dark-1)
            sizes_here = np.array(sorted(sizes_here))
            sizes_spots_dark.append(sizes_here)
            if len(sizes_here)>0:
                size_largest_dark.append(sizes_here[-1])
                num_spots_above_size_threshold_dark.append(len(np.where(np.array(sizes_here)>size_threshold)[0]))
            else:
                size_largest_dark.append(0)
                num_spots_above_size_threshold_dark.append(0)
            nb_blobs_dark_small = len(np.where(small_blob_condition(sizes_here))[0])
            nb_blobs_dark_large = len(np.where(sizes_here<1000000)[0]) - nb_blobs_dark_small - len(np.where(sizes_here<=LOW_THRESH)[0])
            nb_blobs_dark_long = len(np.where(lenghts>70)[0])
            axes[1][2+threshold_idx].imshow(img_dark_binarized, cmap='gray', interpolation = None, vmin=0, vmax=255)
            axes[1][2+threshold_idx].text(10, 36, "%i small area spots"%(nb_blobs_dark_small), fontsize = 12, c="lightgreen")
            axes[1][2+threshold_idx].text(10, 36+36, "%i large area spots (%s)"%(nb_blobs_dark_large, np.sort(sizes_here)[-3:]), fontsize = 12, c="lightgreen")
            axes[1][2+threshold_idx].text(10, 36+2*36, "%i long spots (%s)"%(nb_blobs_dark_long, np.sort(lenghts)[-3:]), fontsize = 12, c="lightgreen")
            axes[1][2+threshold_idx].text(10, 36+3*36, "ratio (%s)"%(white_ratio_dark[-1]), fontsize = 12, c="lightgreen")

        plt.savefig("analysis/images/analysis_%s.png"%(ID), dpi=200)
        plt.close()

        nums_spots_bright_all.append(nums_spots_bright)
        size_largest_bright_all.append(size_largest_bright)
        num_spots_above_size_threshold_bright_all.append(num_spots_above_size_threshold_bright)
        nums_spots_dark_all.append(nums_spots_dark)
        size_largest_dark_all.append(size_largest_dark)
        num_spots_above_size_threshold_dark_all.append(num_spots_above_size_threshold_dark)

        ratio_bright_all.append(white_ratio_bright)
        ratio_dark_all.append(white_ratio_dark)

    nums_spots_bright_all = np.array(nums_spots_bright_all)
    size_largest_bright_all = np.array(size_largest_bright_all)
    num_spots_above_size_threshold_bright_all = np.array(num_spots_above_size_threshold_bright_all)
    nums_spots_dark_all = np.array(nums_spots_dark_all)
    size_largest_dark_all = np.array(size_largest_dark_all)
    num_spots_above_size_threshold_dark_all = np.array(num_spots_above_size_threshold_dark_all)
    num_bins_above_threshold_bright_all = np.array(num_bins_above_threshold_bright_all)
    num_bins_above_threshold_dark_all = np.array(num_bins_above_threshold_dark_all)

    ratio_bright_all = np.array(ratio_bright_all)
    ratio_dark_all = np.array(ratio_dark_all)

    
    feature_names = []
    for size in range(len(thresholds_bright)):
        feature_names.append("num_spots_bright_threshold%i"%(size))
    for size in range(len(thresholds_bright)):
        feature_names.append("size_largest_bright_threshold%i"%(size))
    for size in range(len(thresholds_bright)):
        feature_names.append("num_spots_above_size_threshold_bright_threshold%i"%(size))
    for size in range(len(thresholds_dark)):
        feature_names.append("num_spots_dark_threshold%i"%(size))
    for size in range(len(thresholds_dark)):
        feature_names.append("size_largest_dark_threshold%i"%(size))
    for size in range(len(thresholds_dark)):
        feature_names.append("num_spots_above_size_threshold_dark_threshold%i"%(size))
    
    for size in range(len(thresholds_bright)):
        feature_names.append("ratio_bright%i"%(size))
    for size in range(len(thresholds_dark)):
        feature_names.append("ratio_dark%i"%(size))
    print("Generated %i features"%(len(feature_names)))
    
    features_all = np.hstack([nums_spots_bright_all,
                              size_largest_bright_all,
                              num_spots_above_size_threshold_bright_all,
                              nums_spots_dark_all,
                              size_largest_dark_all,
                              num_spots_above_size_threshold_dark_all,
                              ratio_bright_all,
                              ratio_dark_all,
                              ])
    # print("Final shape of feature matrix: ", features_all.shape)

    data_dict = {}
    for row, ID in enumerate(list_todo):
        feat = features_all[row]
        data_dict[ID] = {n: v.item() for n, v in zip(feature_names, feat)}
    
    with open('./analysis/images/inhomogeneity.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    # np.savetxt("analysis/images/features_all.txt", features_all)
    
    # outfile = open("analysis/images/feature_names.txt", "w")
    # for x in feature_names:
    #     outfile.write("%s\n"%(x))
    # outfile.close()

    # outfile = open("analysis/images/filenames.txt", "w")
    # for x in filenames:
    #     outfile.write("%s\n"%(x))
    # outfile.close()




if __name__=='__main__':
    folder = "data/images"
    analyse_images(folder)
