import cv2
import numpy as np
from IPython.display import clear_output


def display_np_img(img):
    img = img.copy().astype("float")
    img -= img.min()
    img /= img.max()
    img = (255 * img).astype("uint8")

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img


def update(img, x, y):
    tmp = mamm_w_heatmap
    tmp[..., 2] = x * tmp[..., 2] + y * result2
    display_np_img(tmp)


def show_mamm_w_boxes(processed_mamm, prediction, th=0.5):
    result = (np.tile(processed_mamm[..., None], (1, 1, 3)) * 255).astype("uint8")
    bbs = np.zeros_like(result)
    cc = cv2.connectedComponentsWithStats((prediction > th).astype("uint8"), 8)
    for i in range(1, cc[0]):
        start_point = cc[2][i][0] - 5, cc[2][i][1] - 5
        end_point = start_point[0] + cc[2][i][2] + 10, start_point[1] + cc[2][i][3] + 10
        cv2.rectangle(bbs, start_point, end_point, (0, 0, 255), cv2.FILLED)
    clear_output()
    display_np_img(cv2.addWeighted(result, 1.0, bbs, 0.5, 1))


def segmentation_map(prediction, threshold=0.5):
    seg_map = prediction.copy()
    seg_map[seg_map < threshold] = 0
    seg_map[seg_map > 0] = 1

    return seg_map * 255
