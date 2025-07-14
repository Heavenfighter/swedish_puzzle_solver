from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pytesseract

from spellchecker import SpellChecker

class ImageProcessor:

    def __init__(self) -> None:
        self._spellchecker = SpellChecker(language='de')

    def preprocess(self, img) -> np.ndarray:
        """
        Convert image to a binary mask highlighting grid lines.

        Steps:
          1. Grayscale
          2. Gaussian blur
          3. Adaptive threshold (binary inverse)

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Binary image.
        """
        tmp_img = img.copy()

        gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    def find_grid_contour(self, bin_img) -> np.ndarray:
        """
        Find the largest quadrilateral contour assumed to be the puzzle grid.

        Args:
            bin_img (np.ndarray): Binary (thresholded) image.

        Returns:
            np.ndarray: Array of four corner points (x, y).

        Raises:
            ValueError: If no 4-point contour is detected.
        """

        # Find external contours in the binary image
        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area in descending order (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # If the polygon has 4 points, it is likely the puzzle grid
            if len(approx) == 4:
                return approx.reshape(4, 2)

        raise ValueError("No quadrilateral grid contour found.")

    @staticmethod
    def sort_corners(pts):
        """
        Sort four points in the order: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts (np.ndarray): Array of shape (4, 2) with unordered corner coordinates.

        Returns:
            np.ndarray: Sorted corners as float32.
        """
        pts = pts[np.argsort(pts[:, 1])]  # sort by y
        top = pts[:2][np.argsort(pts[:2, 0])]  # sort top two by x
        bottom = pts[2:][np.argsort(pts[2:, 0])]  # sort bottom two by x
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    def warp(self, img, corners) -> np.ndarray:
        """
        Apply perspective transform to get a top-down view of the puzzle.

        Args:
            img (np.ndarray): Original BGR image.
            corners (np.ndarray): Four unordered corner points.

        Returns:
            np.ndarray: Warped BGR image of size self.warp_size.
        """
        rect = self.sort_corners(corners)
        h, w, c = img.shape

        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)

        return cv2.warpPerspective(img, M, (w, h))

    def detect_grid_lines(self, img_warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract horizontal and vertical grid lines via morphology.

        Args:
            img_warped (np.ndarray): Warped BGR image.

        Returns:
            tuple: (horizontal_lines, vertical_lines) binary masks.
        """
        warped = img_warped.copy()
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )

        # lines must be at least 50% size of image
        width, height = bw.shape
        horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.5), 1))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height*0.5)))

        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_k)
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_k)

        return horiz, vert

    def remove_border_artifacts(self, img:np.ndarray,
                                max_thickness:int = 5,
                                thresh:int = 127,
                                max_coverage:float = 0.05,
                                debug:bool = False) -> np.ndarray:
        """
        Remove dark frame artifacts from all four edges by painting inward.
        Uses a displayable binary mask (0 or 255), measures dark‐pixel coverage
        per row/column, and stops when coverage exceeds max_coverage or thickness
        exceeds max_thickness.

        Args:
            img:            Input image (grayscale)
            max_thickness:  Max pixels to paint from each edge.
            thresh:         Grayscale threshold (0–255) to distinguish dark pixels.
            max_coverage:   Upper bound on dark‐pixel fraction for a “thin” artifact.
            debug:          If True, show intermediate binary mask and coverage logs.

        Returns:
            A copy of img with detected border artifacts painted white.
        """

        gray = img.copy()
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold → dark pixels = 255, background = 0
        # Note: THRESH_BINARY_INV turns <= thresh → 255, > thresh → 0
        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

        if debug:
            cv2.imshow("gray", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("binary_mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # normalize mask to {0,1} for coverage calculation
        binary = (mask // 255).astype(np.uint8)
        h, w = binary.shape

        result = img.copy()

        for side in ["top", "bottom", "left", "right"]:

            # est_min_coverage:  calculated Lower bound on dark‐pixel fraction for a “thick” artifact.
            # has to be calculated for every side, because with painting side white,
            # fraction changes
            top_fracs = [binary[r, :].mean() for r in range(min(max_thickness,binary.shape[0]))]
            bot_fracs = [binary[h - 1 - r, :].mean() for r in range(min(max_thickness,binary.shape[0]))]
            left_fracs = [binary[:, c].mean() for c in range(min(max_thickness,binary.shape[1]))]
            right_fracs = [binary[:, w - 1 - c].mean() for c in range(min(max_thickness,binary.shape[1]))]

            all_fracs = top_fracs + bot_fracs + left_fracs + right_fracs

            # pick the *highest* border coverage seen, then back off by 10%
            est_min_coverage = max(all_fracs) - 0.1
            est_min_coverage = max(0.0, est_min_coverage)

            # coverage check helpers
            def row_is_border(y):
                row = binary[y, :]
                frac = row.mean()  # dark‐pixel fraction
                if debug:
                    print(f"Row {y}: frac={frac:.3f}")

                # treat as border if fraction is very low or very high
                return (frac <= max_coverage) or (frac >= est_min_coverage)

            def col_is_border(x):
                col = binary[:, x]
                frac = col.mean()
                if debug:
                    print(f"Col {x}: frac={frac:.3f}")
                return (frac <= max_coverage) or (frac >= est_min_coverage)

            if side == "top":
                for r in range(h):
                    if row_is_border(r):
                        result[r, :] = 255
                        binary[r, :] = 0
                    else:
                        break
            elif side == "bottom":
                for r in range(h):
                    y = h - 1 - r
                    if row_is_border(y):
                        result[y, :] = 255
                        binary[y, :] = 0
                    else:
                        break
            elif side == "left":
                for c in range(w):
                    if col_is_border(c):
                        result[:, c] = 255
                        binary[:, c] = 0
                    else:
                        break
            else:
                for c in range(max_thickness):
                    x = w - 1 - c
                    if col_is_border(x):
                        result[:, x] = 255
                        binary[:, x] = 0
                    else:
                        break

        if debug:
            cv2.imshow("result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    def crop_border(self, img: np.ndarray, remove: int = 4) -> np.ndarray:
        """
        Crops a fixed number of pixels from all four sides of the image.

        Args:
            img:    Input image (2D or 3D NumPy array).
            remove: Number of pixels to remove from each edge (top, bottom, left, right).

        Returns:
            Cropped image as a NumPy array.

        Raises:
            ValueError: If the border is negative or too large for the image size.
        """

        if remove < 0:
            raise ValueError("Border must be >= 0")

        h, w = img.shape[:2]

        if 2 * remove >= h or 2 * remove >= w:
            raise ValueError("Border too large for the image dimensions")

        return img[remove:h - remove, remove:w - remove]

    def remove_disortions(self, img:np.ndarray) -> np.ndarray:
        """
        Removes noise from image

        Args:
          img:       Input image (grayscale).

        Returns:
          image array with reduced noise.
        """
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        thresh = cv2.threshold(blur, 100, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        return invert

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def get_cell_dark_ratio(self, gray_cell_img:np.ndarray) -> int:
        """
        Calculates the ratio of dark (non-white) pixels in the center region of the image.

        The image is thresholded using adaptive thresholding (inverted), and the central
        region (half the size of the image) is analyzed.

        :param gray_cell_img: Grayscale image to inspect
        :return: Ratio of dark pixels in the center, scaled to range 0–1000
        """
        blur = cv2.GaussianBlur(gray_cell_img, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)

        # take only center of image with half of size
        img_size = thresh.shape
        h, w = thresh.shape
        h = h // 2
        w = w // 2

        x = img_size[1] / 2 - w / 2
        y = img_size[0] / 2 - h / 2

        roi = thresh[int(y):int(y + h), int(x):int(x + w)]

        return int(cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1]) * 1000)

    def get_cell_split(self, gray_cell_img:np.ndarray) -> Optional[Dict[str, list[np.ndarray]]]:
        """
        Detects and extracts the main dividing line (if any) in a cell image,
        and splits the cell into two separate regions along that line.

        In Swedish-style puzzles, only horizontal or diagonal dividers are used.
        The first returned part is always the lower or left region (typically the clue),
        and the second part is the upper or right region (typically the solution space).

        :param gray_cell_img: A grayscale image of a puzzle cell.
        :return: A dictionary with the orientation of the split and the resulting image parts.
                 If no divider is found, returns the original image with orientation "none".
        """
        img = gray_cell_img.copy()
        # enlarge for better results
        #img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)

        tmp = self.remove_border_artifacts(img, thresh=240, max_coverage=0.09)
        #tmp = self.crop_border(tmp, remove=3)
        edges = cv2.Canny(tmp, 100, 240, apertureSize=3)

        # line has to be at least 60% of length of the smaller image dimension
        min_len = int(min(img.shape) * 0.6)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                                minLineLength=min_len,
                                maxLineGap=10)
        if lines is None:
            return {
                "orientation": "none",
                "parts": [img]
            }

        best_line = None
        max_len = 0

        # find longest line – assumed to be the main divider
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_len:
                max_len, best_line = length, (x1, y1, x2, y2)

        # no divider found
        if best_line is None:
            return {
                "orientation": "none",
                "parts": [img]
            }

        x1, y1, x2, y2 = best_line

        # get angle of dividing line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # Identify the orientation based on angle (only horizontal or diagonal allowed) 10° deviation allowed
        treshold = 10
        if abs(angle - 0) < treshold or abs(angle - 180) < treshold:
            orientation =  'horizontal'
        elif abs(angle - 45) < treshold:
            orientation = 'diagonal'  # top left to bottom right
        else:
            if abs(angle) == 90:
                # vertical line could be artefact from border
                return {
                    "orientation": "none",
                    "parts": [img]
                }

            cv2.imshow('Edges', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            raise AssertionError("Unsupported cell divider angle: %d" % abs(angle))

        # create mask
        height, width = img.shape
        yy, xx = np.mgrid[0:height, 0:width]
        vals = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
        bottom_left_mask = vals > 0
        top_right_mask = vals < 0

        def crop_region(mask:np.ndarray) -> np.ndarray:
            """
            Extracts and crops the sub-image defined by the given binary mask.
            """
            ys, xs = np.where(mask)
            if ys.size == 0:
                return np.zeros((0, 0, img.shape[2])) if img.ndim == 3 else np.zeros((0, 0),
                                                                                               dtype=img.dtype)
            y0, y1_ = ys.min(), ys.max()
            x0, x1_ = xs.min(), xs.max()
            return img[y0:y1_ + 1, x0:x1_ + 1]

        bottom_left = crop_region(bottom_left_mask) # bottom or left image part
        top_right = crop_region(top_right_mask) # top or right image part

        return {
            "orientation": orientation,
            "parts": [bottom_left, top_right]
        }

    def extract_text(self, row_idx: int, col_idx: int, gray_cell_img:np.ndarray,
                     with_spellcheck=False, debug=False) -> Optional[str]:

        gray = gray_cell_img.copy()

        # remove borders
        without_borders = self.remove_border_artifacts(gray, thresh=240, max_coverage=0.03, max_thickness=5)

        # enlarge for better results
        #enlarged = cv2.resize(without_borders, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # sharpen using unsharp masking
        result = self.unsharp_mask(without_borders, amount=3.0)

        if debug:
            #cv2.imshow('input', gray)
            #cv2.imshow('without_borders', without_borders)
            #cv2.imshow('enlarged', enlarged)
            cv2.imshow('result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        whitelist = "abcdefghijklmnopqrstuvwxyzüöäABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-ÜÖÄß: ="
        custom_config = f'--psm 11 --oem 3'
        data = pytesseract.image_to_data(result, lang='deu', config=custom_config,
                                         output_type=pytesseract.Output.DICT)

        # Instead of using image_to_text, get text from data
        # to eliminate recognized linebreaks
        lines: Dict[int, List[str]] = {}
        for i, w in enumerate(data['text']):
            if not w.strip():
                continue

            ln = data['line_num'][i]
            lines.setdefault(ln, []).append(w)

        clue = ""
        for words in lines.values():
            clue = " ".join(words).replace("- ", "")

        if not clue:
            return None

        approximations = {
            'ō': 'ö',
            'Ō': 'Ö',
            'á': 'ä', 'à': 'ä', 'â': 'ä',
            'ç': 'c',
            'é': 'e', 'è': 'e', 'ê': 'e',
            'ë': 'e',
            'í': 'i', 'ì': 'i',
            'ñ': 'n',
            'ó': 'ö', 'ò': 'ö', 'õ': 'ö',
            'ú': 'ü', 'ù': 'ü',
        }

        for orig, repl in approximations.items():
            clue = clue.replace(orig, repl)

        # filter non ASCII
        clue = "".join(c for c in clue if c in whitelist)

        if with_spellcheck:
            #try spellcheck the word to gain quality
            corrected = []

            for word in clue.split(' '):
                word_clean = ''.join(c for c in word if c.isalnum())
                if 4 > len(word_clean) > 0:
                    corrected.append(word_clean)
                    continue

                if word_clean.lower() not in self._spellchecker:
                    suggestion = self._spellchecker.correction(word_clean)

                    if suggestion:
                        print(f"correcting '{word}' -> '{suggestion}'")
                        corrected.append(suggestion)
                else:
                    corrected.append(word_clean)

            if len(corrected):
                clue = " ".join(corrected)

        return clue if clue else None