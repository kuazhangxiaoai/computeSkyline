import os
import numpy as np
import argparse
import cv2

exsisted_sky_thresh = 150
diff_time_thresh =2.0

def computeAverageRGB(img):
    B_img, G_img, R_img = img[:, :, 0], img[:,:, 1], img[:, :, 2]
    B_Average,G_Average, R_Average = np.mean(B_img),np.mean(G_img),np.mean(R_img)
    return [B_Average,G_Average, R_Average]

def computeAverageGray(img):
    Average = np.mean(img),
    return Average[0]

def compareBlockImage(blocks, type='averageRGB'):
    assert type in ['averageRGB', 'averageGray', 'histogramRGB', 'histogramGray']
    d_values, d_indexes = [],[]
    if type == 'averageRGB':
        n = len(blocks)
        for i in range(n-1):
            f0 = blocks[i]
            f1 = blocks[i+1]
            d_blue = np.abs(f0[0] + 1e-5) / np.abs(f1[0] + 1e-5)
            d_green= np.abs(f0[1] + 1e-5) / np.abs(f1[1] + 1e-5)
            d_red  = np.abs(f0[2] + 1e-5) / np.abs(f1[2] + 1e-5)
            d_value = np.max(np.array([d_blue,d_green,d_red]))
            d_index = np.argmax(np.array([d_blue,d_green,d_red]))
            d_values.append(d_value)
            d_indexes.append(d_index)
        return np.array(d_values),np.array(d_indexes)
    if type == 'averageGray':
        n = len(blocks)
        for i in range(n - 1):
            f0 = blocks[i]
            f1 = blocks[i + 1]
            d_value = np.abs(f0 + 1e-5)/np.abs(f1 + 1e-5)
            d_index = 0
            d_values.append(d_value)
            d_indexes.append(d_index)
        return np.array(d_values), np.array(d_indexes)

def HistogramRGB(img):
    histogram = {
        'blue':[],
        'green': [],
        'red': []
    }
    histogram['blue'] = cv2.calcHist([img], [0], None, 256,[0, 256])
    histogram['green'] = cv2.calcHist([img], [1], None, 256, [0, 256])
    histogram['red'] = cv2.calcHist([img], [2], None, 256, [0, 256])
    return histogram

def computeBottomSkyline(img, bar_size=19, mode='RGB'):
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[0], img.shape[1]
    if mode == 'gray':
        h, w = img.shape[0], img.shape[1]

    bar_num = h // bar_size
    means = []
    for i in range(bar_num):
        bar = img[i * bar_size: (i+1)*bar_size, :]
        mean = np.mean(bar)
        means.append(mean)
    bottomSkyline = bar_size * np.argmin(np.array(means)) + bar_size // 2
    return bottomSkyline

def parse_arg():
    parser = argparse.ArgumentParser(description="compute sky line")
    parser.add_argument('--input-image-path', default=None,help='input image path')
    parser.add_argument('--output-image-path', default=None, help='output image path')
    parser.add_argument('--intput-image-dir', default=None, help='output image directory')
    parser.add_argument('--output-image-dir', default=None, help='output image directory')

    args = parser.parse_args()
    return args


def computeSkyline(img, bar_num=129, block_num=120, mode='RGB'):
    bottomline = computeBottomSkyline(img)
    img = img[:int(bottomline),:,:]
    h,w = img.shape[0], img.shape[1]
    bar_size = w // bar_num
    block_size = h // block_num
    bar_sky_list_y = [0] * bar_num
    bar_sky_list_x = [0] * bar_num

    for i in range(bar_num):
        bar_start_x = i * bar_size
        bar_end_x = (i + 1) * bar_size
        bar_img = img[:, bar_start_x:bar_end_x,:]
        bar_sky_list_x[i] = i * bar_size + bar_size/2
        mean_list = []
        for j in range(block_num):
            block_start_y = j * block_size
            block_end_y = (j + 1) * block_size
            block_img = bar_img[block_start_y: block_end_y, :, :]
            if mode == 'RGB':
                means = computeAverageRGB(block_img)
            if mode == 'gray':
                means = computeAverageGray(block_img[:,:, 0])
            mean_list.append(means)
        judge_color = mean_list[0]
        if mode == 'RGB':
            average_judge_color = 0.333 * (judge_color[0] + judge_color[1] + judge_color[2])
        if mode == 'gray':
            average_judge_color = judge_color
        if average_judge_color < exsisted_sky_thresh:
            bar_sky_list_y[i] = 0
        else:
            diff_values, diff_indexes = compareBlockImage(mean_list) if mode == 'RGB' else compareBlockImage(mean_list,'averageGray')
            sky_line_between_blocks, sky_line_between_blocks_index = np.max(diff_values),np.argmax(diff_values)
            sky_line_between_blocks_chennels = diff_indexes[sky_line_between_blocks_index]
            #for j in range(len(diff_values)):
            #    if diff_values[j] > diff_time_thresh:
            #        sky_line_between_blocks_index = j
            bar_sky_list_y[i] = sky_line_between_blocks_index * block_size

    return bottomline, bar_sky_list_x, bar_sky_list_y



def main():
    arg = parse_arg()
    if (arg.input_image_path is not None) :
        img = cv2.imread(arg.input_image_path)
        h,w = img.shape[0],img.shape[1]
        input_image = img[: h * 2 // 3, : , : ].copy()
        bottomline, skyline_x, skyline_y = computeSkyline(input_image)
        for x,y in zip(skyline_x, skyline_y):
            cv2.circle(img, center=(int(x),int(y)),radius=10, color=(0,0,255), thickness=10)
        cv2.line(img,pt1=(0, bottomline),pt2=(img.shape[1]-1, bottomline), color=(0,255,255),thickness=10)
        cv2.namedWindow("input",0)
        cv2.imshow("input", img)
        cv2.waitKey()

if __name__ == '__main__':
    main()






