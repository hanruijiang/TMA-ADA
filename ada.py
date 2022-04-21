import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2


try:
    
    from osgeo import gdal

    def get_thumbnail(path, core_diameter, num_columns, num_rows):

        from osgeo import gdal

        slide = gdal.Open(path)

        Level0 = slide.GetRasterBand(1)
        
        downsample = list()

        if core_diameter > 0:
            downsample.append(core_diameter / 128)
        if num_columns > 0:
            downsample.append(Level0.XSize / (num_columns * 200))
        if num_rows > 0:
            downsample.append(Level0.YSize / (num_rows * 200))

        downsample = max(1, int(np.floor(min(downsample))))

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
            
        downsample = list()
            
        if core_diameter > 0:
            downsample.append(core_diameter / 128)
        if num_columns > 0:
            downsample.append(img.size[0] / (num_columns * 200))
        if num_rows > 0:
            downsample.append(img.size[1] / (num_rows * 200))

        downsample = max(1, int(np.floor(min(downsample))))
            
        if downsample > 1:
            thumb = img.resize((img.size[0] // downsample, img.size[1] // downsample), Image.NEAREST)
        else:
            thumb = img

        return thumb.convert('RGB'), downsample
    

def get_cores(thumb, core_diameter, downsample, num_columns, num_rows):
    
    hsv = np.asarray(thumb.convert('HSV'))

    th_1, _ = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_OTSU)
    th_2, _ = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_OTSU)
    
    mask = ((hsv[..., 1] > th_1) + (hsv[..., 2] < th_2)).astype('uint8')
    
    if core_diameter <= 0:
    
        core_d = list()

        if num_rows > 0:
            mask_x = mask.sum(axis=1).astype('uint16')
            th_x, _ = cv2.threshold(mask_x, 0, max(mask_x), cv2.THRESH_OTSU)
            core_d.append(sum(mask_x > th_x) / num_rows)

        if num_columns > 0:
            mask_y = mask.sum(axis=0).astype('uint16')
            th_y, _ = cv2.threshold(mask_y, 0, max(mask_y), cv2.THRESH_OTSU)
            core_d.append(sum(mask_y > th_y) / num_columns)

        core_d = int(np.mean(core_d) * 1.2)

        core_diameter = int(core_d * downsample)

    else:

        core_d = int(np.ceil(core_diameter / downsample))
        
    k2 = int(np.ceil(core_diameter / downsample / 64))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    k8 = int(np.ceil(core_diameter / downsample / 16))
    k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k8, k8))

    mask = (mask * 255).astype('uint8')

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k8, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k8, iterations=4)
    
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
    
    return cores, core_diameter, core_d
    
    
def get_array(cores, core_d):

    xs = [x + w / 2 for m, x, y, w, h, area in cores]
    ys = [y + h / 2 for m, x, y, w, h, area in cores]

    cols = list()
    unprocessed = set(range(len(cores)))

    while len(unprocessed) > 0:

        i = min(unprocessed, key=lambda j: xs[j])
        min_x = xs[i]

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

    rows = list()
    unprocessed = set(range(len(cores)))

    while len(unprocessed) > 0:

        i = min(unprocessed, key=lambda j: ys[j])
        min_y = ys[i]

        row = [i]
        unvisited = [j for j in unprocessed if j != i]

        while len(unvisited) > 0:

            j = min(unvisited, key=lambda k: min([abs(xs[k] - xs[l]) for l in row]))

            last = None
            for ref in sorted(row, key=lambda l: abs(xs[j] - xs[l])):
                delta = abs(ys[ref] - ys[j])
                if last is not None and delta > last:
                    break
                if delta < core_d / 2:
                    row.append(j)
                    break
                last = delta

            unvisited.remove(j)

        rows.append(row)
        unprocessed = unprocessed.difference(row)
    
    coords = [[None, None] for _ in range(len(cores))]

    for i, col in enumerate(cols):
        for j in col:
            coords[j][0] = i

    for i, row in enumerate(rows):
        for j in row:
            coords[j][1] = i
        
    array = [[None for _ in range(len(rows))] for _ in range(len(cols))]

    for (c, r), (m, x, y, w, h, area) in zip(coords, cores):

        if array[c][r] is not None:
            x_o, y_o, w_o, h_o = array[c][r]
            x_min, y_min, x_max, y_max = min(x, x_o), min(y, y_o), max(x + w, x_o + w_o), max(y + h, y_o + h_o)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        array[c][r] = [x, y, w, h]        
        
    return array


def standardize_array(array, core_d):
    
    border = core_d // 12

    ws, hs = list(), list()

    for i, row in enumerate(array):
        for j, box in enumerate(row):

            if box is None:
                continue

            x, y, w, h = box

            ws.append(w)
            hs.append(h)

    w_mean, h_mean = int(np.mean(ws)), int(np.mean(hs))

    for i, row in enumerate(array):
        for j, box in enumerate(row):
            
            if box is None:
                continue

            x, y, w, h = box

            up = None if j == 0 else row[j - 1]
            down = None if j == len(row) - 1 else row[j + 1]
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
    
    for i, row in enumerate(array):
        for j, box in enumerate(row):
            
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
                up = row[j - 1]
                if up is not None:
                    y_min = max(y_min, up[1] + up[3] + 2)
                
            if j < len(row) - 1:
                down = row[j + 1]
                if down is not None:
                    y_max = min(y_max, (y_max + down[1] - 2) // 2)

            array[i][j] = x_min, y_min, x_max - x_min, y_max - y_min
            
            
def get_results(array, width, height, downsample):
    
    results = list()

    for i, row in enumerate(array):
        for j, box in enumerate(row):

            QC_pass = box is not None

            up = None if j == 0 else row[j - 1]
            down = None if j == len(row) - 1 else row[j + 1]
            left = None if i == 0 else array[i-1][j]
            right = None if i == len(array) - 1 else array[i+1][j]

            if not QC_pass:

                up_ = None
                for k in range(j - 1, -1, -1):
                    if row[k] is not None:
                        up_ = row[k]
                        break

                down_ = None
                for k in range(j + 1, len(row)):
                    if row[k] is not None:
                        down_ = row[k]
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

                if up is not None:
                    y = max(y, up[1] + up[3] + 2)

                if down is not None:
                    h = min(h, down[1] - y - 2)

                if left is not None:
                    x = max(x, left[0] + left[2] + 2)

                if right is not None:
                    w = min(w, right[0] - x - 2)

            else:

                x, y, w, h = box
                
            x_min, x_max, y_min, y_max = x, x + w, y, y + h
            x_min, x_max, y_min, y_max = max(x_min, 0), min(x_max, width), max(y_min, 0), min(y_max, height)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            results.append([i, j, x * downsample, y * downsample, w * downsample, h * downsample, QC_pass])
            
    results = pd.DataFrame(results, columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass'])
    
    return results


def check_results(results, num_columns, num_rows):

    cols = results.query('QC_pass')['col'].unique()
    if num_columns > 0 and num_columns < len(cols):
        cols = sorted(cols, key=lambda c: results.query(f'QC_pass and col == {c}').shape[0])
        for c in cols[:len(cols) - num_columns]:
            results.drop(results.query(f'col == {c}').index, axis=0, inplace=True)
        for i, c in enumerate(sorted(cols[len(cols) - num_columns:])):
            results.loc[results.query(f'col == {c}').index, 'col'] = i

    rows = results.query('QC_pass')['row'].unique()
    if num_rows > 0 and num_rows < len(rows):
        rows = sorted(rows, key=lambda r: results.query(f'QC_pass and row == {r}').shape[0])
        for r in rows[:len(rows) - num_rows]:
            results.drop(results.query(f'row == {r}').index, axis=0, inplace=True)
        for i, r in enumerate(sorted(rows[len(rows) - num_rows:])):
            results.loc[results.query(f'row == {r}').index, 'row'] = i
    

def visualize(thumb, results, downsample, core_d):
    
    vis = np.asarray(thumb)

    for name, c, r, x, y, w, h, QC_pass in results.itertuples(name=None):
        
        x, y, w, h = x // downsample, y // downsample, w // downsample, h // downsample

        color = (0, 255, 0) if QC_pass else (255, 0, 0)

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness=core_d // 16)

        cv2.putText(vis, name, org=(x + core_d // 6, y + core_d // 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=core_d / 64, color=0, thickness=core_d // 16)
        
    return Image.fromarray(vis)


def de_array(path, core_diameter, num_columns, num_rows):
    
    thumb, downsample = get_thumbnail(path, core_diameter, num_columns, num_rows)
    
    cores, core_diameter, core_d = get_cores(thumb, core_diameter, downsample, num_columns, num_rows)
        
    if len(cores) == 0:
        print(f'ERROR: can not find candidate cores {path}')
        return pd.DataFrame(columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass']), thumb
    
    array = get_array(cores, core_d)
    
    standardize_array(array, core_d)
        
    if len(array) == 0 or len(array[0]) == 0:
        print(f'ERROR: can not group cores into rows and cols {path}')
        return pd.DataFrame(columns=['col', 'row', 'x', 'y', 'w', 'h', 'QC_pass']), thumb
        
    results = get_results(array, thumb.size[0], thumb.size[1], downsample)
        
    valid_cores = results.query('QC_pass').shape[0]
    
    check_results(results, num_columns, num_rows)
    
    cols = results.query('QC_pass')['col'].unique()
    rows = results.query('QC_pass')['row'].unique()

    th_d = min(len(cols), len(rows)) // 2
    if valid_cores - results.query('QC_pass').shape[0] > th_d:
        print(f'WARNING: drop too many cores {path}')
        
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
    