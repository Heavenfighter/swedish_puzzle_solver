# SwedishPuzzleSolver

Tries to solve a swedish styled puzzle from image.

1. [About](#about)
2. [Installation](#installation)
3. [Limitations](#limitations)

## <a name="about"></a>About

**swedish_puzzle_solver** is a small is a console-based application written in Python.
It analyzes a picture of a crossword puzzle and tries to get solutions from various online sources.
These solutions are analyzed by a backtracking algorithm and then inserted into the puzzle.

The result is shown and clues which weren't solved (due incorrect text recognition or where no answers could be found) are highlighted.

## <a name="installation"></a>Installation

### Installation from source

1. The following components need to be installed:
   1. [Python](https://www.python.org/) **3.10** or newer
   1. [pip](https://pypi.org/project/pip/)
   1. [git client](https://git-scm.com/downloads)


1. Open a command/terminal window
1. Clone the repo using
   ```
   git clone https://github.com/Heavenfighter/swedish_puzzle_solver.git
   ```
1. Change into the directory:
   ```
   cd swedish_puzzle_solver
   ```
1. Install the Python dependencies using:
   ```
   pip install pdm

   pdm install
   ```
1. Run the app:
   ```
   pdm run app --help
   ```
## <a name="limitations"></a>Limitations

Since there are many different types of puzzles, the graphical recognition is not always reliable and needs to be improved.
Also the text recognition with tesseract doesn't work for all clues.

Some limitations are:

- supports only german language 
- no puzzles with cut-outs (advertising block or solution block in the grid), only completely rectangular grids
- weird kind of clue arrows aren't supported, only straight arrows (top and down or in 90° angles) 
- minimum resolution of the image 600x600