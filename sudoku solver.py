sudoku = [
    ['.', '.', '6', '.', '.', '1', '7', '.', '5'], 
    ['.', '.', '.', '9', '.', '.', '.', '.', '.'], 
    ['3', '.', '.', '.', '.', '.', '.', '6', '.'], 
    ['.', '4', '.', '.', '.', '6', '2', '.', '.'], 
    ['5', '6', '.', '.', '.', '.', '.', '4', '.'], 
    ['.', '.', '3', '8', '.', '.', '1', '.', '.'], 
    ['.', '7', '.', '.', '.', '9', '.', '8', '.'], 
    ['.', '.', '.', '6', '.', '4', '.', '.', '7'], 
    ['9', '.', '5', '.', '.', '2', '.', '.', '.']
]

for i in range(9):
    for j in range(9):
        if sudoku[i][j]!=".":
            sudoku[i][j] = int(sudoku[i][j])
        else:
            sudoku[i][j] = 0

print("Unsolved puzzle:")
print(*sudoku, sep = "\n", end = "\n\n")

def check_in_col(row, col):
    for i in range(9):
        if i!=row and sudoku[i][col]==sudoku[row][col]:
                return False
    return True

def check_in_row(row, col):
    for j in range(9):
        if j!=col and sudoku[row][j]==sudoku[row][col]:
            return False
    return True

def check_in_box(row, col):
    for i in range(row - row%3, row - row%3 + 3):
        for j in range(col - col%3, col - col%3 + 3):
            if (i!=row or j!=col) and sudoku[i][j]==sudoku[row][col]:
                return False
    return True

def find_empty_space():
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]==0:
                return (i, j)
    return False

def solve_sudoku():
    box = find_empty_space()
    if not box:
        return sudoku


    for i in range(1, 10):
        sudoku[box[0]][box[1]] = i
        if check_in_row(*box) and check_in_col(*box) and check_in_box(*box):
            s = solve_sudoku()
            if not s:
                sudoku[box[0]][box[1]] = 0
            else:
                return s
        else:
            sudoku[box[0]][box[1]] = 0
    return False

print("Solved puzzle:")
print(*solve_sudoku(), sep = "\n", end = "\n\n")

