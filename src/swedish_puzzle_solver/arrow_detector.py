import enum
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np


class ArrowDirection(enum.Enum):
    RIGHT_DOWN = "right_down"
    LEFT_DOWN = "left_down"
    DOWN_RIGHT = "down_right"
    UP_RIGHT = "up_right"
    RIGHT = "right"
    DOWN = "down"

class Arrow:
    def __init__(self, direction: ArrowDirection):
        self.direction = direction

    def __str__(self):
        return self.direction.value if self.direction else ""

    def __repr__(self):
        return self.__str__()

class ArrowDetector:
    VECTOR_MAP = {
        ArrowDirection.RIGHT_DOWN: {
            "source_vector": (0, -1),
            "direction": (1, 0),
        },
        ArrowDirection.LEFT_DOWN: {
            "source_vector": (0, 1),
            "direction": (1, 0),
        },
        ArrowDirection.DOWN_RIGHT: {
            "source_vector": (-1, 0),
            "direction": (0, 1),
        },
        ArrowDirection.UP_RIGHT: {
            "source_vector": (1, 0),
            "direction": (0, 1),
        },
        ArrowDirection.DOWN: {
            "source_vector": (-1, 0),
            "direction": (1, 0),
        },
        ArrowDirection.RIGHT: {
            "source_vector": (0, -1),
            "direction": (0, 1),
        },
    }

    TEMPLATE_MAP = {
        ArrowDirection.DOWN: {
            "template_path": ["res/templates/down.png", "res/templates/down_2.png"],
        },
        ArrowDirection.RIGHT: {
            "template_path": ["res/templates/right.png", "res/templates/right_2.png"],
        },
    }

    def __init__(self, image_gray, scale_factors=None):
        if scale_factors is None:
            scale_factors = [1.0, 0.9, 1.1]

        self.image_gray = image_gray
        self.scales = scale_factors

    def __load_templates(self) -> dict[ArrowDirection, list[np.ndarray]]:
        templates: Dict[Any, List[np.ndarray]] = {}

        for key, value in self.TEMPLATE_MAP.items():
            for path in value["template_path"]:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    raise FileNotFoundError(f"Template '{key}' konnte nicht geladen werden: {path}")

                templates.setdefault(key, []).append(img)

        return templates

    def find_arrows_with_template(self, match_threshold=0.8, min_black_fraction=0.1, debug=False) -> list[Arrow]:
        """
        Detects arrow type vie template matching.
        If no match is found with original scale, other scales are tried.
        Templates are shrinked to cell size.
        Takes two runs:
            - first run complex arrows are searched
            - second simple arrows (right, down) are searched

        Returns:
            List[Arrow]: list of unique arrows
        """
        best_match = None
        best_val = -1
        arrows = []
        run = 1

        all_templates = self.__load_templates()

        # filter "simple" arrows since they are also found in complex arrow types
        template_to_search_for = {k: v for k, v in all_templates.items() if
                                  k not in {ArrowDirection.DOWN, ArrowDirection.RIGHT} }

        while True:
            for direction, template_list in template_to_search_for.items():

                for template in template_list:
                    # shrink template to max cell size
                    height, width = template.shape[:2]
                    if width > self.image_gray.shape[1] or height > self.image_gray.shape[0]:
                        scale_w = self.image_gray.shape[1] / width
                        scale_h = self.image_gray.shape[0] / height

                        # take lower scale ratio
                        scale_ratio = min(scale_w, scale_h, 1.0)

                        new_width = int(width * scale_ratio)
                        new_height = int(height * scale_ratio)

                        template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    for scale in self.scales:

                        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                        # resize is to big for cell_img
                        if resized.shape[0] > self.image_gray.shape[0] or resized.shape[1] > self.image_gray.shape[1]:
                            break

                        res = cv2.matchTemplate(self.image_gray, resized, cv2.TM_CCOEFF_NORMED)
                        h, w = template.shape[:2]
                        loc = np.where(res >= match_threshold)

                        for pt in zip(*loc[::-1]):
                            # Extract the matched region from the image
                            y1, y2 = pt[1], pt[1] + resized.shape[0]
                            x1, x2 = pt[0], pt[0] + resized.shape[1]
                            match_region = self.image_gray[y1:y2, x1:x2]

                            if debug:
                                cv2.imshow("match_region", match_region)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                            # Calculate fraction of black pixels (below threshold, e.g. < 100)
                            black_pixels = np.sum(match_region < 100)
                            total_pixels = match_region.size
                            black_fraction = black_pixels / total_pixels

                            if black_fraction < min_black_fraction:
                                continue  # Too bright -> skip

                            arrows.append(Arrow(direction))

            if len(arrows) or run == 2:
                # complex arrows found or nothing found, were done
                break
            else:
                # now search for "simple" arrows
                run += 1
                template_to_search_for = {k: v for k, v in all_templates.items() if
                                          k in {ArrowDirection.DOWN, ArrowDirection.RIGHT}}

        return self.__filter_unique_directions(arrows)

    def __filter_unique_directions(self, arrows):
        unique = {}
        for arrow in arrows:
            if arrow.direction not in unique:
                unique[arrow.direction] = arrow
        return list(unique.values())

    def __find_arrow_direction(self, contour):
        """
        Determine the arrow’s pointing direction by:
          1) Creating a minimal mask of the filled contour.
          2) Skeletonizing the mask to find thin centerlines.
          3) Locating skeleton endpoints (pixels with only one neighbor).
          4) Selecting the endpoint farthest from the contour centroid as the arrow tip.
          5) Falling back to the farthest contour point if no endpoints are detected.
        """
        # 1. Compute bounding box to limit mask size and speed up processing
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Shift contour coordinates so they fit into the mask
        cnt_shifted = contour.copy()
        cnt_shifted[:, :, 0] -= x
        cnt_shifted[:, :, 1] -= y

        # Draw the filled contour into the mask
        cv2.drawContours(mask, [cnt_shifted], -1, 255, thickness=-1)

        # 2. Attempt skeletonization via OpenCV contrib; fallback to raw mask
        try:
            skeleton = self.__skeletonize(mask)
        except AttributeError:
            skeleton = mask.copy()

        # 3. Find skeleton endpoints (pixels with exactly one neighbor)
        endpoints = []
        pts = np.argwhere(skeleton > 0)
        for r, c in pts:
            # Extract 3×3 neighborhood around the pixel
            window = skeleton[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            # Count non-zero pixels (itself + neighbors)
            if np.count_nonzero(window) == 2:
                # Append endpoint in global coordinates
                endpoints.append((c + x, r + y))

        # 4. Compute contour centroid for fallback and tip selection
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 5. Choose arrow tip:
        #    - If skeleton endpoints exist, pick the one farthest from centroid.
        #    - Otherwise, pick the farthest contour point from centroid.
        if endpoints:
            # Compute squared distances and pick max
            tipx, tipy = max(endpoints, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        else:
            # Fallback to raw contour points
            dists = [((pt[0][0] - cx) ** 2 + (pt[0][1] - cy) ** 2, pt[0])
                     for pt in contour]
            _, (tipx, tipy) = max(dists, key=lambda x: x[0])

        #debug_img = cv2.cvtColor(self.image_gray, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(debug_img, contour, -1, (0, 255, 0), 1)
        #cv2.circle(debug_img, (cx, cy), 4, (255, 0, 0), -1)  # centroid blue
        #cv2.circle(debug_img, (tipx,tipy), 4, (0, 255, 0), -1)  # tip green
        #for pt in endpoints:
        #    cv2.circle(debug_img, pt, 2, (0, 0, 255), -1)  # endpoints red

        #cv2.imshow("Arrow Direction Detection Advanced", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # 6. Derive cardinal direction from vector (dx, dy)
        dx, dy = tipx - cx, tipy - cy
        if abs(dx) > abs(dy):
            return (1, 0) if dx > 0 else None
        else:
            return (0, 1) if dy > 0 else None

    def __create_mask_from_contour(self, cnt, shape):
        """
        Draws a filled mask of a single contour in its minimal bounding-box ROI.
        Returns the mask and the top-left offset (x,y) of the ROI.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_shift = cnt.copy()
        cnt_shift[:, :, 0] -= x
        cnt_shift[:, :, 1] -= y
        cv2.drawContours(mask, [cnt_shift], -1, 255, thickness=-1)
        return mask, (x, y)

    def __skeletonize(self, img_bin):
        """
        Skeletonize binary image with morphological thinning (Zhang-Suen-artig).
        Input img_bin: 0/255 binary image.
        """
        img = img_bin.copy() // 255
        skel = np.zeros(img.shape, np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel * 255

    def detect_arrow_threshold(self):
        # adaptive binary threshold (inverted)
        bw = cv2.adaptiveThreshold(self.image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

        # Close small gaps to connect arrow fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external contours in the binary image
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrows = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # filter out tiny or huge blobs (gt then 80 percent of cell)
            if area < 100 or area > (0.8 * self.image_gray.shape[0] * self.image_gray.shape[1]):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            direction = None
            # If polygon has 5–12 vertices, try skeleton-based tip detection
            if 4 <= len(approx) <= 12:
                direction = self.__find_arrow_direction(cnt)

            if direction:
                print( "down" if direction == (0,1) else "right")
                # If a direction was found, compute the target cell
                #off_col, off_row = direction
                #target_col, target_row = row_idx + off_col, col_idx + off_row
                #for dir_str, dir in self.SOURCE_VECTOR.items():
                #    if dir == direction:
                #       arrows.append(Arrow(str(dir_str)))
                #       break
        return arrows

    def detect_black_lines_near_edges(self, image, threshold=127,
                                      tolerance=0, debug=False) -> Tuple[List[str], Dict[str, bool]]:
        """
        Detects on which side(s) of a grayscale image black lines are present near the border.
        Searches for the first black pixel from the edge inwards, and selects the side(s)
        with the smallest distance to the black pixel found. If multiple sides are within
        the tolerance range of the smallest distance, they are all considered valid.

        :param image: Grayscale image (np.ndarray)
        :param threshold: Pixel intensity below which a pixel is considered "black"
        :param tolerance: Tolerance range for valid distances (e.g., 2 means sides within
                          min_distance + 2 are also considered valid)
        :param debug: If True, display a debug image with markings
        :return: A list of detected sides (e.g., ['top', 'right']) or empty list if none found
        """

        if len(image.shape) != 2 or image.dtype != np.uint8:
            raise ValueError("Input must be a grayscale image.")

        height, width = image.shape
        original_image = image.copy()

        blur = cv2.GaussianBlur(image, (3, 3), 0)

        _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)

        # store the distances to the first black pixel found
        distances = {'top': None, 'bottom': None, 'left': None, 'right': None}

        # Search for the first black pixel on each side and calculate the distance
        # Top: Search rows from top to bottom
        for y in range(height):
            if np.any(mask[y, :] == 255):  # Find if any pixel in the row is black
                distances['top'] = y
                break

        # Bottom: Search rows from bottom to top
        for y in range(height - 1, -1, -1):  # Start from the bottom row
            if np.any(mask[y, :] == 255):  # Find if any pixel in the row is black
                distances['bottom'] = height - 1 - y
                break

        # Left: Search columns from left to right
        for x in range(width):
            if np.any(mask[:, x] == 255):  # Find if any pixel in the column is black
                distances['left'] = x
                break

        # Right: Search columns from right to left
        for x in range(width - 1, -1, -1):  # Start from the right column
            if np.any(mask[:, x] == 255):  # Find if any pixel in the column is black
                distances['right'] = width - 1 - x
                break

        # determine the smallest distance(s)
        valid_distances = {side: dist for side, dist in distances.items() if dist is not None}

        if debug:
            cv2.imshow(f'Debug', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if not valid_distances:
            return [], {side: False for side in distances}

        min_distance = min(valid_distances.values())

        # set the tolerance range for valid distances (min_distance <= dist <= min_distance + tolerance)
        valid_sides = [side for side, dist in valid_distances.items() if
                       min_distance <= dist <= min_distance + tolerance]

        # dictionary indicating if the side was detected (True or False)
        side_flags = {side: side in valid_sides for side in distances}

        if debug:
            vis = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR)
            overlay = vis.copy()
            alpha = 0.4
            color = (0, 0, 255)

            if side_flags['top']:
                cv2.rectangle(overlay, (0, 0), (width, 10), color, -1)
            if side_flags['bottom']:
                cv2.rectangle(overlay, (0, height - 10), (width, height), color, -1)
            if side_flags['left']:
                cv2.rectangle(overlay, (0, 0), (10, height), color, -1)
            if side_flags['right']:
                cv2.rectangle(overlay, (width - 10, 0), (width, height), color, -1)

            debug_img = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
            cv2.imshow(f'Debug', debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return valid_sides, side_flags