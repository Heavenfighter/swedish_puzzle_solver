# SwedishPuzzleSolver
Tries to solve a swedish styled puzzle from image.

This is a small program written in Python.
It analyzes a picture of a crossword puzzle and tries to get solutions from various online sources.
These solutions are analyzed by a backtracking algorithm and then inserted into the puzzle.

Since there are different ways of constructing the puzzles, the graphical recognition is unstable and needs to be improved.

Some limitations are
- no puzzles with cut-outs (advertising block or solution block in the grid), only completely rectangular grids
- minimum resolution of the image 600x600
