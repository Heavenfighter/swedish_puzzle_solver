
from __future__ import annotations

import enum
import os
import sys
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

#from spellchecker import SpellChecker
from swedish_puzzle_solver.arrow_detector import ArrowDetector, Arrow, ArrowDirection
from swedish_puzzle_solver.image_processor import ImageProcessor
from swedish_puzzle_solver.online_solver import OnlineSolver

class Celltype(enum.Enum):
    EMPTY = enum.auto()
    ARROW = enum.auto()
    CLUE = enum.auto()

# BGR colors
TYPE_COLORS = {
    Celltype.EMPTY: (50, 50, 50),
    Celltype.CLUE: (255, 0, 0),
    Celltype.ARROW: (0, 0, 255),
}

class Cell:
    """
    One grid cell, which may carry:
      - an arrow (and thus a clue index/text)
      - a text‐cell (clue number printed inside, rare in Schwedenrätsel)
      - an empty fill cell
    """

    def __init__(self, board: Board, row: int, col: int, image: np.ndarray):
        self.row = row
        self.col = col
        self.board = board

        self.top_left = [0, 0]
        self.bottom_right = [0, 0]

        self.image = image
        self.letter = None  # filled letter

        self.type = Celltype.EMPTY

    def is_fillable(self) -> bool:
        return isinstance(self, LetterCell)

    def set_position(self, top_left: [int, int], bottom_right: [int, int]):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def set_type(self, type: Celltype):
        self.type = type

    def has_arrow(self):
        return isinstance(self, LetterCell) and self.has_arrow

class ClueCell(Cell):
    def __init__(self, board:Board, row:int, col:int, image:np.ndarray):
        super().__init__(board, row, col, image)

        # a clue cell can contain up to two clues
        self.clues: List[Clue] = []
        self.type = Celltype.CLUE
        self._solver = OnlineSolver()

    def lookup_answers(self, debug=False):
        for clue in self.clues:
            clue.add_candidates(self._solver.lookup_answers_online(clue.id, len(clue.path), debug))
            #print(f'self.board._slots_by_id["{clue.id}"].add_candidates({clue.candidates})')

            if debug:
                print(clue.candidates)

    def add_clue(self, text:str):
        clue = Clue(text, self)
        self.clues.append(clue)

class Clue:
    def __init__(self, text:str, cell:ClueCell):
        self.id = text
        self.candidates: List[str] = []
        self.path: List[Tuple[int, int]] = []
        self.clue_cell = cell

        # used for MRV:
        self.remaining: List[str] = []

        # filled in build_neighbour_map()
        # id -> (my_index, their_index)
        self.neighbours: Dict[str, Tuple[int, int]] = {}

    @property
    def length(self) -> int:
        return len(self.path)

    def add_candidates(self, word:List[str]):
        self.candidates.extend(word)

        # take all candidates as remaining
        self.remaining = self.candidates.copy()

class LetterCell(Cell):
    def __init__(self, board: Board, row:int, col:int, image:np.array):
        super().__init__(board, row, col, image)

        self.has_arrow: bool = False
        self.char: Optional[str] = None
        self.arrows: list[Arrow] = []

    def add_arrows(self, arrows: list[Arrow]):
        self.has_arrow = True
        self.arrows.extend(arrows)
        self.type = Celltype.ARROW

class Board:
    """
    The full puzzle: warped image, matrix of Cells, and clue‐objects.
    """

    def __init__(self, warped: np.array ):
        self.warped = warped
        self.cells: List[List[Cell]] = [[]]

    def cell_at(self, row: int, col: int) -> Optional[Cell]:
        if 0 <= row < len(self.cells) and 0 <= col < len(self.cells[0]):
            return self.cells[row][col]
        return None

    def fill_letter(self, r: int, c: int, ch: str):
        self.cells[r][c].letter = ch

    @property
    def clues(self) -> List[Clue]:
        return [clue for sublist in self.cells for cell in sublist if isinstance(cell, ClueCell) for clue in cell.clues]

    @property
    def clue_cells(self) -> List[ClueCell]:
        return [cell for sublist in self.cells for cell in sublist if isinstance(cell, ClueCell)]

    @property
    def letter_cells(self) -> List[LetterCell]:
        return [cell for sublist in self.cells for cell in sublist if isinstance(cell, LetterCell)]

    @property
    def arrow_cells(self) -> List[LetterCell]:
        return [cell for sublist in self.cells for cell in sublist if isinstance(cell, LetterCell) and cell.has_arrow]

class SwedishPuzzleSolver(ImageProcessor):
    """
    Detects grid with OpenCV, OCRs clues, queries online, then backtracks fill.
    """

    def __init__(self) -> None:
        super().__init__()

        self.__orig_img = None
        self.__warped_img = None
        self.board: Board

    @staticmethod
    def __cluster_coords(coords:List[int], eps:int=10) -> List[int]:
        """
        Cluster 1D integer coordinates that lie close to each other (within 'eps')
        and return the mean coordinate of each cluster.

        Args:
            coords (list or np.ndarray): A list of integer coordinates (should be sortable).
            eps (int): Maximum allowed distance between two values to be grouped into the same cluster.

        Returns:
            list of int: A list of mean coordinates, one for each cluster.
        """

        # ensure coordinates are sorted before clustering
        coords = sorted(coords)

        # initialize the first cluster with the first coordinate
        clusters = [[coords[0]]]

        for c in coords[1:]:
            #  if the current coordinate is close enough to the last coordinate in the current cluster
            if abs(c - clusters[-1][-1]) < eps:
                clusters[-1].append(c) # add
            else:
                clusters.append([c]) # new cluster

        # return the mean value of each cluster
        return [int(np.mean(cl)) for cl in clusters]

    def __extract_cells(self, img_warped:np.ndarray):
        horiz, vert = self.detect_grid_lines(img_warped)
        inters = cv2.bitwise_and(horiz, vert)

        ys, xs = np.where(inters > 0)
        rows = self.__cluster_coords(list(ys))
        cols = self.__cluster_coords(list(xs))

        self.board.cells.clear()
        for row_idx, (y1, y2) in enumerate(zip(rows[:-1], rows[1:])):
            row = []
            for col_idx, (x1, x2) in enumerate(zip(cols[:-1], cols[1:])):
                img_cell = img_warped[y1:y2, x1:x2]

                cell = self.__classify_cell(row_idx, col_idx, img_cell)
                cell.set_position((x1, y1), (x2, y2))

                row.append(cell)
            self.board.cells.append(row)

        self.board.slots_by_id = {c.id: c for c in self.board.clues}

    def __classify_cell(self, row_idx: int, col_idx: int, cell_img:np.ndarray) -> Cell:
        """
        Classify a single cell as 'text', 'arrow_<dir>', or 'empty'.

        - Uses Tesseract OCR for text detection.
        - Detects arrows in a single cell:
          1) Adaptive thresholding + morphological closing.
          2) Contour extraction and area filtering.
          3) Polygon approximation → if vertex count is within [5,12], use skeleton-based tip detection.
          4) Fallback: skeleton endpoints on mask to estimate tip direction.
          5) Final fallback: template matching if provided.
          6) Compute target cell based on detected direction.

        Args:
            row_idx: index of row containing the cell
            col_idx: index of row containing the cell
            cell_img (np.ndarray): BGR image of one cell.

        Returns:
            Cell: new cell with proper type
        """

        print(f'analyse: row {row_idx} col {col_idx}')

        # Convert to grayscale
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # pre-classify with white ratio
        dark_ratio = self.get_cell_dark_ratio(gray)
        print(f'\tdark_ratio: {dark_ratio}')

        cell = LetterCell(self.board, row=row_idx, col=col_idx, image=cell_img)
        if dark_ratio > 125:
            # should be clue cell

            # resize for better ocr and split detection
            resized_up = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
            res = self.get_cell_split(resized_up)

            cell = ClueCell(self.board, row_idx, col_idx, cell_img)

            for gray_part_img in res["parts"]:

                #cv2.imshow('cell part', gray_part_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                if col_idx == 0 and row_idx == 9:
                    pass

                text = self.extract_text(row_idx, col_idx, gray_part_img)
                if text:
                    cell.add_clue(text)
                    print(f'\tclues: {text}')
                else:
                    cell.add_clue("s")

            if len(cell.clues) != len(res["parts"]):
                # not all clues could be determined
                print("Could not determine all clues")
        else:
            enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

            # Arrow detection via template matching
            detector = ArrowDetector(enlarged)
            arrows = detector.find_arrows_with_template(match_threshold=0.87, min_black_fraction=0.05)

            if row_idx == 0 and col_idx == 2:
                pass

            if len(arrows):
                without_borders = self.remove_border_artifacts(gray, thresh=240, max_coverage=0.03)
                without_borders = self.crop_border(without_borders, remove=2)

                resized_up = cv2.resize(without_borders, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)

                # if img is cropped by crop_border search_depth only 2, else 5
                # depends on thickness of outer lines.
                # Some puzzles have an small distance between cell border an arrow
                # maybe an TODO: improve detection
                #_, arrow_source_sides_dict = (
                 #   detector.detect_black_lines_near_edges_old(image=resized_up,
                #                                           threshold=150,
                #                                           edge_search_depth=6,
                #                                           debug=(row_idx==4 and col_idx==1)))

                # treshold 127 for cells with light circles
                _, arrow_source_sides_dict = (
                    detector.detect_black_lines_near_edges(image=resized_up,
                                                           threshold=170, tolerance=2))

                if not any(arrow_source_sides_dict.values()) or sum(1 for value in arrow_source_sides_dict.values() if value) != len(arrows):
                    raise ValueError("arrow source could not be determined!")

                # with two arrows in one cell, we have to conclude the right direction
                # arrow[0]: always down
                # arrow[1]: always right
                if len(arrows) > 1:
                    # von rechts gehts nur nach unten
                    if arrow_source_sides_dict['right']:
                        # kann nur down sein
                        arrows[0].direction = ArrowDirection.LEFT_DOWN
                    elif arrow_source_sides_dict['bottom']:
                        # von unten gehts nur nach rechts
                        arrows[1].direction = ArrowDirection.UP_RIGHT

                    if arrow_source_sides_dict['top']:
                        # entweder nach unten oder nach unten rechts
                        if not arrow_source_sides_dict['left']:
                            # kann nur nach unten rechts sein
                            arrows[1].direction = ArrowDirection.DOWN_RIGHT
                else:
                    # one arrow and one side
                        if arrows[0].direction == ArrowDirection.RIGHT:
                            # arrow points right, can only occur from bottom or top
                            if arrow_source_sides_dict['bottom']:
                                arrows[0].direction = ArrowDirection.UP_RIGHT
                            elif arrow_source_sides_dict['top']:
                                arrows[0].direction = ArrowDirection.DOWN_RIGHT
                        else:
                            #arrow points down, can only occur from left or right
                            if arrow_source_sides_dict['left']:
                                arrows[0].direction = ArrowDirection.RIGHT_DOWN
                            elif arrow_source_sides_dict['right']:
                                arrows[0].direction = ArrowDirection.LEFT_DOWN

                cell.add_arrows(arrows)
                print(f'\tArrows: {arrows}')

        return cell

    def __build_paths_and_attach(self, board: Board):
        """
        1. For each arrow cell, trace its fill‐path
        2. Assign clue_idx & clue_text to that cell
        3. Record path in Clue
        """
        idx = 1
        for cell in board.arrow_cells:
            for arr_idx, arrow in enumerate(cell.arrows):
                #clue_pos = tuple(
                #    x + y for x, y in zip((cell.row, cell.col), ArrowDetector.TEMPLATE_MAP[arrow.direction]["source_vector"]))

                source_vec = ArrowDetector.VECTOR_MAP[arrow.direction]["source_vector"]
                clue_pos = tuple((cell.row + source_vec[0], cell.col + source_vec[1]))

                clue_cell = board.cell_at(clue_pos[0], clue_pos[1])
                if not clue_cell:
                    continue

                dir_row, dir_col = ArrowDetector.VECTOR_MAP[arrow.direction]["direction"]

                path = [(cell.row, cell.col)]
                nr, nc = cell.row + dir_row, cell.col + dir_col

                while 0 <= nr < len(board.cells) and 0 <= nc < len(board.cells[0]) and board.cells[nr][nc].is_fillable():
                    path.append((nr, nc))
                    nr, nc = nr + dir_row, nc + dir_col

                if len(clue_cell.clues) > 1:
                    if source_vec[0] == 0:
                        clue_cell.clues[1].path = path
                    else:
                        clue_cell.clues[0].path = path
                else:
                    clue_cell.clues[0].path = path

                idx += 1

    def build_board(self, image_path:str) -> None:

        # Check if file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Error: The file '{image_path}' does not exist.")

        # Check if the file is readable
        if not os.access(image_path, os.R_OK):
            raise ValueError(f"Error: The file '{image_path}' is not readable. Check file permissions.")

        self.__orig_img = cv2.imread(image_path)
        binary = self.preprocess(self.__orig_img)

        corners = self.find_grid_contour(binary)
        self.__warped_img = self.warp(self.__orig_img, corners)

        self.board = Board(warped=self.__warped_img)

        # build board and classify all cells
        self.__extract_cells(self.__warped_img)

        self.print_grid(img=self.__warped_img, cells=self.board.cells, debug=True)

    @staticmethod
    def print_grid(img:np.ndarray, cells:List[List[Cell]], debug=False):
        """
        Zeichnet alle Zellen auf eine Kopie von `image` und
        gibt das markierte Bild zurück.
        """
        vis = img.copy()

        if len(vis.shape) == 2 or vis.shape[2] == 1:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for row in cells:
            for cell in row:

                # fillcolor depends on type
                color = TYPE_COLORS.get(cell.type, (128, 128, 128))

                clue_error = False
                if isinstance(cell, ClueCell) and any(not clue.candidates for clue in cell.clues) and not debug:
                    clue_error = True
                    color = (0,0,255)

                # fill background for classification
                roi = vis[cell.top_left[1]:cell.bottom_right[1], cell.top_left[0]:cell.bottom_right[0]]
                overlay = roi.copy()
                overlay[:] = color
                alpha = 0.5

                if debug or clue_error:
                    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

                if debug:
                    cv2.rectangle(
                        vis,
                        cell.top_left,
                        cell.bottom_right,
                        (0, 255, 0),
                        1
                    )

                    # show cell indices
                    cv2.putText(
                        vis,
                        f"{cell.row},{cell.col}",
                        (cell.top_left[0] + 3, cell.top_left[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                    # draw arrow name
                    if isinstance(cell, LetterCell) and cell.has_arrow:

                        text = ""
                        for arrow in cell.arrows:

                            if text:
                                text += ", "

                            text += arrow.direction.value

                        # calculate orig text size
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                              fontScale=0.3, thickness=1)

                        cv2.putText(vis, text,
                                    (cell.top_left[0] + 3, cell.bottom_right[1] - text_height - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 0),
                                    thickness=1,
                                    lineType=cv2.LINE_AA)

                if cell.letter:
                    img_height, img_width = cell.image.shape[:2]

                    # calculate orig text size
                    (text_width, text_height), baseline = cv2.getTextSize(cell.letter, cv2.FONT_HERSHEY_SIMPLEX,
                                                                          fontScale=1, thickness=3)

                    # get scale 70%
                    scale = (0.7 * img_height) / text_height

                    # calculate scaled font size
                    (text_width_scaled, text_height_scaled), baseline = cv2.getTextSize(cell.letter,
                                                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                                                        fontScale=scale,
                                                                                        thickness=3)

                    # get centered text position
                    x = cell.top_left[0] + ((img_width - text_width_scaled) // 2)
                    y = cell.top_left[1] + ((img_height + text_height_scaled) // 2)

                    cv2.putText(vis, cell.letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=(0, 0, 0),
                                thickness=3,
                                lineType=cv2.LINE_AA)

        # cv2.imwrite("debug_cells.png", vis)
        cv2.imshow("Board", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def __build_neighbour_map(self, clues:List[Clue]) -> None:
        """
        Populate the .neighbours dictionaries of every clue.

        neighbours[other_id] = (my_index, their_index)

        Meaning:
          • my_index      – index *inside my own word* that touches the crossing cell
          • their_index   – index *inside the other word* that touches the same cell

        After this function returns, each pair of crossing slots has two mirror
        entries, one in each direction, so the solver can jump either way.
        """

        cell_map: Dict[Tuple[int, int], Tuple[str, int]] = {}  # remembers the *first* clue that occupied each grid‐coordinate
        slots_by_id = {s.id: s for s in clues} #lookup table for clues

        for clue in clues:
            for idx, cell in enumerate(clue.path):
                if cell in cell_map:
                    # cell already used by another clue
                    other_clue_id, other_idx = cell_map[cell]

                    # add a *bidirectional* neighbour entry
                    #   earlier clue .neighbours[this_id]  = (other_index, my_index)
                    #   current clue .neighbours[other_id] = (my_index,  other_index)
                    slots_by_id[other_clue_id].neighbours[clue.id] = (other_idx, idx)
                    clue.neighbours[other_clue_id] = (idx, other_idx)
                else:
                    cell_map[cell] = (clue.id, idx)

    def __fits_fast(self, word:str, clue:Clue, assignment:Dict[str, str]) -> bool:
        """
        Return True  ->  'word' is compatible with all *already assigned* neighbours
        Return False ->  at least one crossing letter conflicts

        Important:
          We check *only* the crossing cells, never the whole word.
          That makes this constant‑time per crossing and avoids reading / writing the grid structure at all.
        """
        for other_id, (my_idx, their_idx) in clue.neighbours.items():

            # neighbour is not assigned yet
            if other_id not in assignment:
                continue

            # check if crossing letter matches
            if word[my_idx] != assignment[other_id][their_idx]:
                return False

        return True

    def __compatible_words(self, clue:Clue, assignment:Dict[str, str]) -> List[str]:
        # collect only those words that don’t conflict with any neighbour
        return [word for word in clue.remaining if self.__fits_fast(word, clue, assignment)]

    def __select_next_clue(self, clues:List[Clue],
                           assignment:Dict[str, str]) -> Optional[Clue]:
        """
        Pick the *next* clue to branch on, using an MRV (Minimum‑Remaining‑Values) heuristic,
        while ignoring clues whose current domain is empty.

        ─ Parameters ─
        clues       : all clue objects in the puzzle
        assignment  : {clue_id -> word} mapping of already fixed clues

        ─ Returns ─
        The chosen clue (the most constrained one that still has ≥1 compatible words),
        or None if *no* unassigned clue has any compatible words left.
        """
        best: Optional[Clue] = None
        best_len = float('inf') # smallest domain size seen so far

        for clue in clues:
            if clue.id in assignment:
                continue

            comp = self.__compatible_words(clue, assignment)
            if not comp:
                continue

            # cache the filtered domain so the caller can reuse it without recomputing
            clue._cmp_cache = comp

            # Minimum Remaining Values: pick the smallest domain
            domain_size  = len(comp)
            if (domain_size  < best_len or
                    (domain_size  == best_len and best is not None and
                     len(clue.neighbours) > len(best.neighbours))):
                best, best_len = clue, domain_size

        return best

    def solve(self, debug=False):

        # link each Clue to its arrow‐cell & build paths
        self.__build_paths_and_attach(self.board)

        if debug:
            slots_by_id = {s.id: s for s in self.board.clues}  # lookup table for clues
            # for debug test answers fpr schwede_2.jpg
            slots_by_id["japanische Sportart"].add_candidates(['BUDO', 'JUDO', 'SUMO'])
            slots_by_id["Nordafrikaner"].add_candidates(
                ['AEGYPTER', 'ALGERIER', 'ERITREER', 'FELLACHE', 'SUDANESE', 'TUNESIER'])
            slots_by_id["sumpfig"].add_candidates(['MOORLAND', 'MORASTIG'])
            slots_by_id["Abk: für Logarithmus"].add_candidates(['LOG'])
            slots_by_id["Ausruf der Überraschung"].add_candidates(['AH', 'HO', 'OH'])
            slots_by_id["aufgeben"].add_candidates(['ABTUN', 'ENDEN', 'GEHEN'])
            slots_by_id["altes Antriebswerk"].add_candidates(['TRETRAD'])
            slots_by_id["Tier, das frisst"].add_candidates([])
            slots_by_id["ausklammern"].add_candidates(
                ['AUSGENZEN', 'AUSLASSEN', 'AUSNEHMEN', 'AUSSPAREN', 'STREICHEN', 'WEGLASSEN'])
            slots_by_id["Kfz von Esslingen"].add_candidates(['ES', 'NT'])
            slots_by_id["Instrument"].add_candidates(
                ['FEILE', 'GERAT', 'HILFE', 'ORGEL', 'PAUKE', 'PIANO', 'WAFFE', 'ZIACH'])
            slots_by_id["durchgekocht"].add_candidates(['GAR'])
            slots_by_id["Fürst von Venedig"].add_candidates(['DOGE'])
            slots_by_id["von hier an"].add_candidates(['AB'])
            slots_by_id["Anhöhe"].add_candidates(['BERG', 'HANG', 'NOCK'])
            slots_by_id["ausgelassen"].add_candidates(
                ['ALBERN', 'HEITER', 'JOVIAL', 'LUSTIG', 'MUNTER', 'TOLLEN'])
            slots_by_id["Reitersitz"].add_candidates(['SATTEL'])
            slots_by_id["Bücherfreund"].add_candidates(['LESER'])
            slots_by_id["in der Nähe"].add_candidates(['BEI', 'NAH'])
            slots_by_id["Vortragender"].add_candidates(
                ['DOZENT', 'KUSTOS', 'LEHRER', 'LEITER', 'LEKTOR', 'REDNER', 'SOLIST'])
            slots_by_id["Stadt in der Türkei"].add_candidates(
                ['ABANA', 'ADALA', 'ADANA', 'ADENA', 'AFSIN', 'AFYON', 'AHLAT', 'AIDIN', 'AKCAY', 'AKKUS', 'ALACA',
                 'ARSIN', 'AWANA', 'AYDIN', 'BAFRA', 'BAHCE', 'BANAZ', 'BAYAT', 'BELEK', 'BELEN', 'BESNI', 'BIGHA',
                 'BUCAK', 'BURSA', 'CAMAS', 'CEHAN', 'CESME', 'CINAR', 'CIZRE', 'CORLU', 'CORUM', 'CUMRA', 'DATCA',
                 'DEDIM', 'DEMRE', 'DERIK', 'DIDIM', 'DINAR', 'ERBAA', 'ERCIS', 'ERDEK', 'ERHAC', 'ERZIN', 'ESKIL',
                 'EZINE', 'FATSA', 'GEBZE', 'GEDIZ', 'GERZE', 'GEVAS', 'GEYVE', 'HAVZA', 'IGDIR', 'ILGIN', 'ILICA',
                 'INECE', 'IRESI', 'ISMID', 'ISMIR', 'ISMIT', 'IZMIR', 'IZMIT', 'IZNIK', 'KAHTA', 'KAMAN', 'KAZAN',
                 'KEMER', 'KEPEZ', 'KESAN', 'KILIS', 'KINIK', 'KOESK', 'KONIA', 'KONYA', 'KOZAN', 'KOZLU', 'KUMRU',
                 'MADEN', 'MARAS', 'MILAS', 'MUCUR', 'MUGLA', 'NIGDE', 'NIZIP', 'PAYAS', 'PAZAR', 'SARAY', 'SASON',
                 'SEHIR', 'SEMUN', 'SERIK', 'SIIRT', 'SIMAV', 'SINOP', 'SIVAS', 'SIWAS', 'SOEKE', 'SUHUT', 'SURUC',
                 'TAVAS', 'TERME', 'TOKAT', 'TOSYA', 'UDINE', 'UENYE', 'YOMRA'])
            slots_by_id["Anführer"].add_candidates(['LEADER', 'LEITER', 'MACHER'])
            slots_by_id["feucht"].add_candidates(['NASS'])
            slots_by_id["ab jener Zeit"].add_candidates(['SEITDEM'])
            slots_by_id["Besengriff"].add_candidates(['STIEL'])
            slots_by_id["ängstlich"].add_candidates(
                ['BENAUT', 'GEHMMT', 'MULMIG', 'MUTLOS', 'TIMIDE', 'UNFREI'])
            slots_by_id["Abk. designatus"].add_candidates(['DES'])
            slots_by_id["Stadt in den USA"].add_candidates(
                ['ATLANTA', 'AUGUSTA', 'BEDFORD', 'BISMARK', 'BRISTOL', 'BUFFALO', 'CHATHAM', 'CHELSEA', 'CHESTER',
                 'CHICAGO', 'CLAYTON', 'CLINTON', 'CONCORD', 'DETROIT', 'HOBOKEN', 'HOUSTON', 'JACKSON', 'KAYWEST',
                 'KEYWEST', 'LINCOLN', 'MADISON', 'MEMPHIS', 'MILFORD', 'MILFORT', 'MOBOILE', 'NEWBURG', 'NEWPORT',
                 'NEWYORK', 'NORFOLK', 'OAKLAND', 'OLYMPIA', 'PHOENIX', 'PONTIAC', 'RALEIGH', 'READING', 'SANJOSE',
                 'SANTAFE', 'SEATTLE', 'SPOCANE', 'SPOKANE', 'STLOUIS', 'TAMPICO', 'TEMPICO', 'TRENTON', 'VENTURA',
                 'WICHITA', 'YONKERS'])
            slots_by_id["ägyptische Gottheit"].add_candidates(['AA', 'AM', 'AS', 'HA', 'HU', 'RA', 'RE'])
            slots_by_id["in Ordnung"].add_candidates(['IO', 'JA', 'OK'])
            slots_by_id["Briefhülle"].add_candidates(['KUVERT'])
            slots_by_id["römischer Schutzgeist"].add_candidates(['GENIUS'])
            slots_by_id["feierlicher Brauch"].add_candidates(['RITUAL'])
            slots_by_id["14. Buchstabe"].add_candidates(['N'])
            slots_by_id["Denkvermögen"].add_candidates(['GEIST', 'GRIPS', 'LOGIK', 'RATIO'])
            slots_by_id["3efreier"].add_candidates([])
            slots_by_id["3alkanyewohner"].add_candidates([])
            slots_by_id["Eiform"].add_candidates(['EIIG', 'OVAL'])
            slots_by_id["biblische Sta"].add_candidates([])
            slots_by_id["Appetit"].add_candidates(['BOCK', 'GIER', 'LUST'])
            slots_by_id["Autozubehör"].add_candidates(
                ['AIRBAG', 'BREMSE', 'CARDAN', 'GASZUG', 'KABINE', 'KARDAN', 'KLAPPE', 'KOLBEN', 'LENKER', 'NOTRAD',
                 'RAHMEN', 'REIFEN', 'RELAIS', 'RELING', 'STEUER', 'VENTIL', 'VOLANT', 'WINKER'])
            slots_by_id["Kurzform: in dem"].add_candidates(['IM'])
            slots_by_id["Ansammlung"].add_candidates(
                ['BANDE', 'HALDE', 'HAUFE', 'HERDE', 'HORDE', 'LAGER', 'MASSE', 'MENGE', 'MEUTE', 'ROTTE', 'RUDEL',
                 'SCHAR'])
            slots_by_id["Kristallom"].add_candidates([])
            slots_by_id["Brillenbehälter"].add_candidates(['ETUI'])
            slots_by_id["altrömische Münze"].add_candidates(['AS'])
            slots_by_id["dickköpfig"].add_candidates(['STUR'])
            slots_by_id["20. Buchstabe"].add_candidates(['T'])
            slots_by_id["Enzym zur Käseherstellung"].add_candidates(['LAB'])
            slots_by_id["leblos"].add_candidates(['TOT'])
            slots_by_id["altgriechische Grabsäule"].add_candidates(['STELE'])

        else:
            # lookup candidates online
            for clue_cell in self.board.clue_cells:
                clue_cell.lookup_answers(True)

        for clue in self.board.clues:
            # take all candidates as remaining
            clue.remaining = clue.candidates.copy()

        self.__build_neighbour_map(self.board.clues)

        sol = self.__solve_backtrack(self.board.clues, debug)
        if sol is None:
            return False

        # fill in answers
        for clue in self.board.clues:
            if clue.id in sol:
                word = sol[clue.id]
                for (r, c), ch in zip(clue.path, word):
                    self.board.fill_letter(r, c, ch)

        return True

    def __solve_backtrack(self, clues:List[Clue], debug=False) -> Optional[Dict[str, str]]:
        """
        Backtracking solver that tries to find the assignment
        which fills the maximum number of slots.

        Returns:
            best: a dictionary mapping slot ids to chosen words
                  for the best partial (or complete) solution found.
        """
        best: Dict[str, str] = {}  # global best

        def dfs(assignment:Dict[str, str], depth=0, debug=False) -> None:
            nonlocal best

            if len(assignment) > len(best):
                best = assignment.copy()
                if debug:
                    print(f'{" " * depth}new best assignment ({len(best)}/{len(clues)})')

            # Filter open clues: not assigned and with remaining candidates
            open_clues = [c for c in clues if c.id not in assignment and c.remaining]
            if not open_clues:
                return

            # Select the next slot to assign (using MRV heuristic)
            clue = self.__select_next_clue(open_clues, assignment)
            if clue is None:
                return

            words = clue._cmp_cache

            # Sort candidates by least constraining value (LCV)
            words.sort(key=lambda word: self.__lcv_score(word, clue, clues, assignment))

            for word in words:
                if debug:
                    print(f'{"  " * depth}try {word}')

                assignment[clue.id] = word
                if debug:
                    print(f'{"  " * depth}ok')

                dfs(assignment, depth + 1)

                if debug:
                    print(f'{"  " * depth}{word} dismissed')

                del assignment[clue.id]  # Backtrack

        # start depth-first search
        dfs({}, debug=debug)

        return best

    def __lcv_score(self, word:str, clue:Clue, clues: List[Clue], assignment:Dict[str, str]) -> int:
        """
        Compute the Least Constraining Value (LCV) score for a given candidate word
        in a particular clue.

        The score estimates how much this word restricts the possible words
        in neighboring clues that intersect with this clue.

        A lower score means the word is less constraining (better),
        and should be tried earlier in the search.

        Parameters:
        - word: the candidate word being evaluated for the current slot
        - clue: the current clue object where the word is considered
        - clues: list of all clue-objects in the puzzle
        - assignment: dictionary of current assignments {slot_id: word}

        Returns:
        - score: integer representing how constraining the word is;
                 higher means more constraining
        """
        score = 0

        # Build a mapping from slot IDs to slot objects for quick lookup
        clues_by_id = {c.id: c for c in clues}

        # Iterate over all neighbours of this slot
        # neighbours is assumed to be a dict mapping neighbour clue_id to
        # a tuple (index_in_neighbour, index_in_current_slot)
        for neighbor_id, (neighbor_idx, clue_idx) in clue.neighbours.items():
            neighbor = clues_by_id[neighbor_id]

            if neighbor_id in assignment:
                continue

            # Count how many words in neighbor's remaining candidates
            # are compatible with 'word' at the intersecting position
            compatible_count = 0
            for candidate in neighbor.remaining:
                if neighbor_idx >= len(candidate):
                    continue

                if clue_idx >= len(word):
                    continue

                # Check if the character at neighbor_idx matches the character
                # at slot_idx in 'word' to ensure they fit together
                if candidate[neighbor_idx] == word[clue_idx]:
                    compatible_count += 1

            # The more incompatible candidates are ruled out,
            # the higher the score (more constraining)
            eliminated = len(neighbor.remaining) - compatible_count
            score += eliminated

        return score

def main(args: list[str]) -> None:
    # result_path = Path("test")
    # result_path.mkdir(exist_ok=True)

    solver = SwedishPuzzleSolver()

    os.putenv("TESSDATA_PREFIX", "res")

    solver.build_board("test/schwede_4.jpg")

    if not solver.solve(debug=False):
        print("No solution found.")
        return

    solver.print_grid(img=solver.board.warped, cells=solver.board.cells)

if __name__ == "__main__":
    main(sys.argv)
