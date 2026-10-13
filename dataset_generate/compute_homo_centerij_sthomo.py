import cv2
import numpy as np
def compute_homo_centerij_sthomo(scenter, tcenter, homography_st, crop_size):
    tprime_x, tprime_y = tcenter
    x0, y0 = int(scenter[0] - crop_size // 2), int(scenter[1] - crop_size // 2)
    x1, y1 = int(tprime_x - crop_size // 2), int(tprime_y - crop_size // 2)
    src_pts = np.array([
        [x0, y0],
        [x0 + crop_size-1, y0],
        [x0 + crop_size-1, y0 + crop_size-1],
        [x0, y0 + crop_size-1]
    ], dtype=np.float32)
    dst_pts = cv2.perspectiveTransform(np.array([src_pts], dtype=np.float32), homography_st)[0]
    dst_pts -= np.array([[x1, y1]], dtype=np.float32)
    sub_homo, _ = cv2.findHomography(src_pts, dst_pts)
    return sub_homo
def compute_homo_stcenter(scenter, tcenter, homograhpy_st, crop_size):
    subh, subw = crop_size
    T_s = np.array([[1, 0, -scenter[0] + subw//2],
                    [0, 1, -scenter[1] + subh//2],
                    [0, 0, 1]], dtype=np.float64)
    T_t = np.array([[1, 0, -tcenter[0] + subw//2],
                    [0, 1, -tcenter[1] + subh//2],
                    [0, 0, 1]], dtype=np.float64)
    sub_homo = np.dot(np.dot(T_t, homograhpy_st.astype(np.float64)), np.linalg.inv(T_s))
    return sub_homo
