import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2


def get_downsample(width, height, core_diameter, num_columns, num_rows):
    
    downsample = list()

    if core_diameter > 0:
        downsample.append(core_diameter / 128)
    if num_columns > 0:
        downsample.append(width / (num_columns * 200))
    if num_rows > 0:
        downsample.append(height / (num_rows * 200))

    return max(1, int(np.floor(min(downsample))))


try:
    
    from osgeo import gdal

    def get_thumbnail(path, core_diameter, num_columns, num_rows):

        from osgeo import gdal

        slide = gdal.Open(path)

        Level0 = slide.GetRasterBand(1)
        
        downsample = get_downsample(Level0.XSize, Level0.YSize, core_diameter, num_columns, num_rows)

        level = -1
        for i in range(Level0.GetOverviewCount()):
            if float(Level0.XSize) / Level0.GetOverview(i).XSize > downsample * 1.05:
                break
            level = i

        overview = Level0.GetOverview(level) if level > -1 else Level0

        thumb = np.zeros((overview.YSize, overview.XSize, 3), dtype='uint8')
        thumb[..., 0] = overview.ReadAsArray(0, 0, overview.XSize, overview.YSize)
        Level0 = slide.GetRasterBand(2)
        overview = Level0.GetOverview(level) if level > -1 else Level0
        thumb[..., 1] = overview.ReadAsArray(0, 0, overview.XSize, overview.YSize)
        Level0 = slide.GetRasterBand(3)
        overview = Level0.GetOverview(level) if level > -1 else Level0
        thumb[..., 2] = overview.ReadAsArray(0, 0, overview.XSize, overview.YSize)
        thumb = Image.fromarray(thumb, mode='RGB')

        resize = float(Level0.XSize) / overview.XSize / downsample
        if resize != 1:
            thumb = thumb.resize((int(thumb.size[0] * resize), int(thumb.size[1] * resize)), Image.NEAREST)

        return thumb.convert('RGB'), downsample
    
except:
    
    def get_thumbnail(path, core_diameter, num_columns, num_rows):

        img = Image.open(path)
        
        downsample = get_downsample(img.size[0], img.size[1], core_diameter, num_columns, num_rows)
            
        if downsample > 1:
            thumb = img.resize((img.size[0] // downsample, img.size[1] // downsample), Image.NEAREST)
        else:
            thumb = img

        return thumb.convert('RGB'), downsample
    
    
def get_mask(thumb, core_diameter, downsample, num_columns, num_rows):
    
    hsv = np.asarray(thumb.convert('HSV'))

    th_1, _ = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_OTSU)
    th_2, _ = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_OTSU)

    valid = (hsv[..., 2] > 16) * (hsv[..., 1] > 16)
    mask = (((hsv[..., 1] > th_1) + (hsv[..., 2] < th_2)) * valid).astype('uint8')
    
    # estimate core diameter and spacing
    
    if core_diameter <= 0:
    
        mask_x = mask.sum(axis=1).astype('uint16')
        mask_y = mask.sum(axis=0).astype('uint16')
    
        core_d = list()

        if num_rows > 0:
            th_x, _ = cv2.threshold(mask_x, 0, max(mask_x), cv2.THRESH_OTSU)
            core_d.append(sum(mask_x > th_x) / num_rows)

        if num_columns > 0:
            th_y, _ = cv2.threshold(mask_y, 0, max(mask_y), cv2.THRESH_OTSU)
            core_d.append(sum(mask_y > th_y) / num_columns)

        core_d = int(np.mean(core_d) * 1.2)

        core_diameter = int(core_d * downsample)
        
        spacing = list()
        
        if num_rows > 0:
            
            valid_w, last_start, last_end = 0, -1, -1
            for i, valid in enumerate(mask_x > th_x):
                if valid:
                    if last_start == -1:
                        last_start = i
                    last_end = i
                else:
                    if i - last_end > core_d * 2:
                        valid_w += last_end - last_start + 1
                        last_start, last_end = -1, -1
            if last_start != -1:
                valid_w += last_end - last_start + 1
                
            spacing.append(valid_w / num_rows)

        if num_columns > 0:
            
            valid_h, last_start, last_end = 0, -1, -1
            for i, valid in enumerate(mask_y > th_y):
                if valid:
                    if last_start == -1:
                        last_start = i
                    last_end = i
                else:
                    if i - last_end > core_d * 2:
                        valid_h += last_end - last_start + 1
                        last_start, last_end = -1, -1
            if last_start != -1:
                valid_h += last_end - last_start + 1
                
            spacing.append(valid_h / num_columns)
            
        spacing = max(0, min(spacing) - core_d)
        n_close = min(2, max(1, int(np.floor(spacing / (core_d / 8)))))
        n_open = max(1, 4 - int(np.round(spacing / (core_d / 8))))

    else:

        core_d = int(np.ceil(core_diameter / downsample))
        n_close = 3
        n_open = 2
        
    # adjust mask
    
    k2 = int(np.ceil(core_diameter / downsample / 64))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    k8 = int(np.ceil(core_diameter / downsample / 16))
    k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k8, k8))
    k16 = int(np.ceil(core_diameter / downsample / 8))
    k16 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k16, k16))

    mask = (mask * 255).astype('uint8')
    
    ## drop isolated pixels
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    
    ## close & fill holes
        
    mask = cv2.dilate(mask, k8, iterations=n_close)
    
    o = 0
    while True:
        if mask[o, o] == 0:
            break
        if mask[o, mask.shape[1] - o - 1] == 0:
            break
        if mask[mask.shape[0] - o - 1, o] == 0:
            break
        if mask[mask.shape[0] - o - 1, mask.shape[1] - o - 1] == 0:
            break
        o += 1

    mask_for_fill = mask.copy()
    cv2.floodFill(mask_for_fill, np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8), (o, o), 255)
    cv2.floodFill(mask_for_fill, np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8), (mask.shape[1] - o - 1, o), 255)
    cv2.floodFill(mask_for_fill, np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8), (o, mask.shape[0] - o - 1), 255)
    cv2.floodFill(mask_for_fill, np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8), (mask.shape[1] - o - 1, mask.shape[0] - o - 1), 255)
    mask_for_fill = cv2.bitwise_not(mask_for_fill)
    mask_for_fill = cv2.dilate(mask_for_fill, k2, iterations=1)
    mask = mask | mask_for_fill

    mask = cv2.erode(mask, k8, iterations=n_close)
    
    ## close
        
    mask = cv2.erode(mask, k16, iterations=n_open)
    mask = cv2.dilate(mask, k16, iterations=min(n_open, 3))
    
    return mask, core_diameter, core_d


def get_cores(mask, core_d):
    
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

    th_min = core_d * 0.55
    th_max = core_d / 0.55
    
    cores = list()
    for i, (x, y, w, h, area) in enumerate(stats):

        if w < th_min or h < th_min or w > th_max or h > th_max or area < th_min ** 2:
            continue

        m = labels == i
        if m.mean() > 0.1:
            continue

        cores.append([m, x, y, w, h, area])
    
    return cores
    
    
def get_array(cores, core_d):

    xs = [x + w / 2 for m, x, y, w, h, area in cores]
    ys = [y + h / 2 for m, x, y, w, h, area in cores]
    
    # assign columns

    cols = list()
    unprocessed = set(range(len(cores)))

    while len(unprocessed) > 0:

        i = min(unprocessed, key=lambda j: xs[j])

        col = [i]
        unvisited = [j for j in unprocessed if j != i]

        while len(unvisited) > 0:

            j = min(unvisited, key=lambda k: min([abs(ys[k] - ys[l]) for l in col]))

            last = None
            for ref in sorted(col, key=lambda l: abs(ys[j] - ys[l])):
                delta = abs(xs[ref] - xs[j])
                if last is not None and delta > last:
                    break
                if delta < core_d / 2:
                    col.append(j)
                    break
                last = delta

            unvisited.remove(j)

        cols.append(col)
        unprocessed = unprocessed.difference(col)
    
    # assign rows

    rows = list()
    unprocessed = set(range(len(cores)))

    while len(unprocessed) > 0:

        i = min(unprocessed, key=lambda j: ys[j])

        col = [i]
        unvisited = [j for j in unprocessed if j != i]

        while len(unvisited) > 0:

            j = min(unvisited, key=lambda k: min([abs(xs[k] - xs[l]) for l in col]))

            last = None
            for ref in sorted(col, key=lambda l: abs(xs[j] - xs[l])):
                delta = abs(ys[ref] - ys[j])
                if last is not None and delta > last:
                    break
                if delta < core_d / 2:
                    col.append(j)
                    break
                last = delta

            unvisited.remove(j)

        rows.append(col)
        unprocessed = unprocessed.difference(col)
        
    # assign coords
    
    coords = [[None, None] for _ in range(len(cores))]

    for i, col in enumerate(cols):
        for j in col:
            coords[j][0] = i

    for i, col in enumerate(rows):
        for j in col:
            coords[j][1] = i
            
    # assign array
        
    array = [[None for _ in range(len(rows))] for _ in range(len(cols))]

    for (c, r), (m, x, y, w, h, area) in zip(coords, cores):

        if array[c][r] is not None:
            x_o, y_o, w_o, h_o = array[c][r]
            x_min, y_min, x_max, y_max = min(x, x_o), min(y, y_o), max(x + w, x_o + w_o), max(y + h, y_o + h_o)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        array[c][r] = [x, y, w, h]        
        
    return array


def get_neighbors(array, i, j):
    
    up = None if j == 0 else array[i][j - 1]
    down = None if j == len(array[i]) - 1 else array[i][j + 1]
    left = None if i == 0 else array[i-1][j]
    right = None if i == len(array) - 1 else array[i+1][j]

    up_ = None
    for k in range(j - 1, -1, -1):
        if array[i][k] is not None:
            up_ = array[i][k]
            break

    down_ = None
    for k in range(j + 1, len(array[i])):
        if array[i][k] is not None:
            down_ = array[i][k]
            break

    left_ = None
    for k in range(i - 1, -1, -1):
        if array[k][j] is not None:
            left_ = array[k][j]
            break

    right_ = None
    for k in range(i + 1, len(array)):
        if array[k][j] is not None:
            right_ = array[k][j]
            break
        
    return up, down, left, right, up_, down_, left_, right_


def adjust_array(array, core_d):
    
    border = core_d // 12
    
    # estimate width & height

    ws, hs = list(), list()

    for i, col in enumerate(array):
        for j, box in enumerate(col):

            if box is None:
                continue

            x, y, w, h = box

            ws.append(w)
            hs.append(h)

    w_mean, h_mean = int(np.mean(ws)), int(np.mean(hs))
    
    # align cores

    for i, col in enumerate(array):
        for j, box in enumerate(col):
            
            if box is None:
                continue

            x, y, w, h = box

            up = None if j == 0 else col[j - 1]
            down = None if j == len(col) - 1 else col[j + 1]
            left = None if i == 0 else array[i-1][j]
            right = None if i == len(array) - 1 else array[i+1][j]

            x_min, x_max = x, x + w

            if up is not None:
                if down is not None:
                    x_min = min(x_min, (up[0] + down[0]) // 2)
                    x_max = max(x_max, (up[0] + up[2] + down[0] + down[2]) // 2)
                else:
                    x_min = min(x_min, up[0])
                    x_max = max(x_max, up[0] + up[2])
            elif down is not None:
                x_min = min(x_min, down[0])
                x_max = max(x_max, down[0] + down[2])

            if left is not None:
                x_min = max(x_min, left[0] + left[2] + border * 2)
            x_min = min(x, x_min)

            x_max = max(x_max, x_min + w_mean)
            if right is not None:
                x_max = min(x_max, right[0] - border * 3)
            x_max = max(x + w, x_max)

            y_min, y_max = y, y + h

            if left is not None:
                if right is not None:
                    y_min = min(y_min, (left[1] + right[1]) // 2)
                    y_max = max(y_max, (left[1] + left[3] + right[1] + right[3]) // 2)
                else:
                    y_min = min(y_min, left[1])
                    y_max = max(y_max, left[1] + left[3])
            elif right is not None:
                y_min = min(y_min, right[1])
                y_max = max(y_max, right[1] + right[3])

            if up is not None:
                y_min = max(y_min, up[1] + up[3] + border * 2)
            y_min = min(y, y_min)

            y_max = max(y_max, y_min + h_mean)
            if down is not None:
                y_max = min(y_max, down[1] - border * 3)
            y_max = max(y + h, y_max)

            array[i][j] = x_min, y_min, x_max - x_min, y_max - y_min
            
    # extend border
    
    for i, col in enumerate(array):
        for j, box in enumerate(col):
            
            if box is None:
                continue
                
            x, y, w, h = box
            
            x_min, x_max, y_min, y_max = x - border, x + w + border, y - border, y + h + border
            
            if i > 0:
                left = array[i-1][j]
                if left is not None:
                    x_min = max(x_min, left[0] + left[2] + 2)
                
            if i < len(array) - 1:
                right = array[i+1][j]
                if right is not None:
                    x_max = min(x_max, (x_max + right[0] - 2) // 2)

            if j > 0:
                up = col[j - 1]
                if up is not None:
                    y_min = max(y_min, up[1] + up[3] + 2)
                
            if j < len(col) - 1:
                down = col[j + 1]
                if down is not None:
                    y_max = min(y_max, (y_max + down[1] - 2) // 2)

            array[i][j] = x_min, y_min, x_max - x_min, y_max - y_min


def check_array(array, core_d, num_columns, num_rows):
    
    # drop unexpected columns & rows
    
    dropped = list()

    if num_columns > 0 and len(array) > num_columns:
        cols = sorted(range(len(array)), key=lambda c: len([1 for i in array[c] if i is not None]))
        for c in sorted(cols[:len(cols) - num_columns], reverse=True):
            dropped.extend(array.pop(c))

    if num_rows > 0 and len(array[0]) > num_rows:
        rows = sorted(range(len(array[0])), key=lambda r: len([1 for col in array if col[r] is not None]))
        for r in sorted(rows[:len(rows) - num_rows], reverse=True):
            for col in array:
                dropped.append(col.pop(r))
                
    # drop empty columns & rows
                
    for i in range(len(array)-1, -1, -1):
        if len([1 for box in array[i] if box is not None]) == 0:
            array.pop(i)
    
    for i in range(len(array[0])-1, -1, -1):
        if len([1 for k in range(len(array)) if array[k][i] is not None]) == 0:
            for col in array:
                col.pop(i)
                
    # rearrange dropped cores

    for i, col in enumerate(array):
        for j, box in enumerate(col):
            
            if box is not None:
                continue

            up, down, left, right, up_, down_, left_, right_ = get_neighbors(array, i, j)
            
            x_min, x_max = list(), list()
            if up_ is not None:
                x_min.append(up_[0])
                x_max.append(up_[0] + up_[2])
            if down_ is not None:
                x_min.append(down_[0])
                x_max.append(down_[0] + down_[2])
            x_min, x_max = min(x_min) - core_d // 4, max(x_max) + core_d // 4
            
            y_min, y_max = list(), list()
            if left_ is not None:
                y_min.append(left_[1])
                y_max.append(left_[1] + left_[3])
            if right_ is not None:
                y_min.append(right_[1])
                y_max.append(right_[1] + right_[3])
            y_min, y_max = min(y_min) - core_d // 4, max(y_max) + core_d // 4

            matches = list()
            for k in range(len(dropped) - 1, -1, -1):

                if dropped[k] is None:
                    dropped.pop(k)
                    continue

                x, y, w, h = dropped[k]
                if x >= x_min and x + w <= x_max and y >= y_min and y + h <= y_max:
                    matches.append(dropped.pop(k))

            if len(matches) > 0:

                x = min([m[0] for m in matches])
                y = min([m[1] for m in matches])
                w = max([m[0] + m[2] for m in matches]) - x
                h = min([m[1] + m[3] for m in matches]) - y

                array[i][j] = (x, y, w, h)
            
            
def get_results(array, width, height, downsample):
    
    results = list()

    for i, col in enumerate(array):
        for j, box in enumerate(col):

            QC_pass = box is not None

            if not QC_pass:
                
                # estimate missing cores

                up, down, left, right, up_, down_, left_, right_ = get_neighbors(array, i, j)

                if up_ is None:
                    x, w = down_[0], down_[2]
                else:
                    x, w = up_[0], up_[2]
                    if down_ is not None:
                        x = (x + down_[0]) // 2
                        w = (w + down_[2]) // 2

                if left_ is None:
                    y, h = right_[1], right_[3]
                else:
                    y, h = left_[1], left_[3]
                    if right_ is not None:
                        y = (y + right_[1]) // 2
                        h = (h + right_[3]) // 2

                if down is not None:
                    h = min(h, down[1] - y - 2)

                if up is not None:
                    y = max(y, up[1] + up[3] + 2)

                if right is not None:
                    w = min(w, right[0] - x - 2)

                if left is not None:
                    x = max(x, left[0] + left[2] + 2)

            else:

                x, y, w, h = box
                
            # adjust box
                
            x_min, x_max, y_min, y_max = x, x + w, y, y + h
            x_min, x_max, y_min, y_max = max(x_min, 0), min(x_max, width), max(y_min, 0), min(y_max, height)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            results.append([i, j, x * downsample, y * downsample, w * downsample, h * downsample, QC_pass])
            
    results = pd.DataFrame(results, columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass'])
    
    return results
    

def visualize(thumb, results, downsample, core_d):
    
    vis = np.asarray(thumb)

    for name, c, r, x, y, w, h, QC_pass in results.itertuples(name=None):

        color = (0, 255, 0) if QC_pass else (255, 0, 0)
        
        x, y, w, h = x // downsample, y // downsample, w // downsample, h // downsample

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness=core_d // 16)

        cv2.putText(vis, name, org=(x + core_d // 6, y + core_d // 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=core_d / 64, color=0, thickness=core_d // 16)
        
    return Image.fromarray(vis)


def de_array(path, core_diameter, num_columns, num_rows):
    
    thumb, downsample = get_thumbnail(path, core_diameter, num_columns, num_rows)
    
    mask, core_diameter, core_d = get_mask(thumb, core_diameter, downsample, num_columns, num_rows)
    
    cores = get_cores(mask, core_d)
    
    if len(cores) == 0:
        print(f'ERROR: can not find candidate cores {path}')
        return pd.DataFrame(columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass']), thumb
    
    array = get_array(cores, core_d)

    adjust_array(array, core_d)
    
    valid_cores = sum([len([1 for i in col if i is not None]) for col in array])
    
    check_array(array, core_d, num_columns, num_rows)
    
    if num_columns > 0 and num_columns != len(array):
        print(f'WARNING: number of detected columns can\'t match input parameter {path} {len(array)} != {num_columns}')
    if num_rows > 0 and num_rows != len(array[0]):
        print(f'WARNING: number of detected rows can\'t match input parameter {path} {len(array[0])} != {num_rows}')
        
    th_d = min(len(array), len(array[0])) // 2
    if valid_cores - sum([len([1 for i in col if i is not None]) for col in array]) > th_d:
        print(f'WARNING: drop too many cores {path}')
        
    if len(array) == 0 or len(array[0]) == 0:
        print(f'ERROR: can not group cores into rows and cols {path}')
        return pd.DataFrame(columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass']), thumb
        
    results = get_results(array, thumb.size[0], thumb.size[1], downsample)
        
    results['name'] = [f'{chr(c + 65)}{r + 1}' for c, r in zip(results['col'], results['row'])]
    results.set_index('name', inplace=True)
        
    vis = visualize(thumb, results, downsample, core_d)
        
    return results, vis



if __name__ == '__main__':
    
    import os
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path',
        type=str,
        default='/path_to_TMA_img/',
        help='TMA image path.'
    )
    parser.add_argument(
        '-d', '--core_diameter',
        type=int,
        default=-1,
        help='core diameter in pixels.'
    )
    parser.add_argument(
        '-c', '--num_columns',
        type=int,
        default=-1,
        help='number of columns.'
    )
    parser.add_argument(
        '-r', '--num_rows',
        type=int,
        default=-1,
        help='number of rows.'
    )

    args, _ = parser.parse_known_args()
    
    results, vis = de_array(args.path, args.core_diameter, args.num_columns, args.num_rows)
    
    results.to_csv(os.path.splitext(args.path)[0] + '.csv')
    
    vis.save(os.path.splitext(args.path)[0] + '.de-array.jpg', quality=50)
    