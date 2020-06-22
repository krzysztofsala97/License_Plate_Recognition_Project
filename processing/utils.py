import numpy as np
import cv2


def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')
    # Load reference hu moments for every sign
    reference_Hu = np.load('processing/ref_hu.npy', allow_pickle='TRUE').item()
    license_number = ""
    work_copy = image.copy()
    # Preprocessing
    hsv = cv2.cvtColor(work_copy.copy(), cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 127])
    upper_black = np.array([180, 50, 255])
    mask = cv2.inRange(hsv.copy(), lower_black, upper_black)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate = np.zeros(work_copy.shape, np.uint8)
    smallest_width = image.shape[1]
    # Finding license plate
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if 45 > rect[2] > -45:
            (w_r, h_r) = rect[1]
        else:
            (h_r, w_r) = rect[1]

        if w_r > work_copy.shape[1] / 3 and 0.32 > h_r / w_r > 0.15:
            if w_r < smallest_width:
                smallest_width = w_r
                smallest_height = h_r
                selected_cnt = box
    cv2.fillPoly(plate, [selected_cnt], (255, 255, 255))
    masked_image = cv2.bitwise_and(mask.copy(), plate[:, :, 0])
    contours2, hierarchy2 = cv2.findContours(masked_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letters = {}
    # Finding letters
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if 45 > rect[2] > -45:
            (w_r, h_r) = rect[1]
        else:
            (h_r, w_r) = rect[1]
        if h_r > 0.6 * smallest_height and w_r < 0.5 * smallest_width:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            letters[cX] = cnt
    num = 0
    for key in sorted(letters):
        M = cv2.moments(letters[key])
    # https://www.docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#humoments
    # Hu Moments are  invariants to the image scale, rotation, and reflection (last one can have different sign)
        Hu = cv2.HuMoments(M)
        for i in range(0, 7):
            Hu[i] = Hu[i] * 10 ** (-1 * (int(np.log10(abs(Hu[i]))) - 1))
        distance = None
        # Comparing contours hu moments with reference dictionery
        for letter in reference_Hu:
            check_dist = 0
            for i in range(0, 7):
                check_dist += (abs(reference_Hu[letter][i]) - abs(Hu[i])) ** 2
            if distance is None or distance > check_dist:
                # First letter is assigned based on voivodeship
                if num == 0 and letter in ["Z", "G", "N", "B", "P", "C", "W", "L",
                                           "F", "D", "O", "S", "K", "R", "T", "E"]:
                    distance = check_dist
                    min_key = letter
                # Second letter can't be a number
                elif num == 1 and letter not in ["0", "1", "2", "3", "4",
                                                 "5", "6", "7", "8", "9"]:
                    distance = check_dist
                    min_key = letter
                elif num > 1:
                    distance = check_dist
                    min_key = letter
        if len(license_number) < 7:
            license_number += min_key
        num += 1
    while len(license_number) < 7:
        license_number += "?"
    return license_number
