import random
from copy import deepcopy

# multiple solutions allowed or not (algo will find the shortest path among them)
mult_sols_allowed = 1

height = 10
width = height
start_point = [1, 1]
end_point = [10, 10]

def create_maze(maze):
    dfs = [start_point]
    while dfs:
        maze[dfs[-1][0]][dfs[-1][1]] = 0
        poss = []
        if dfs[-1][0]>1 and maze[dfs[-1][0]-2][dfs[-1][1]]==1:
            poss.append([dfs[-1][0]-2, dfs[-1][1]])
        if dfs[-1][1]>1 and maze[dfs[-1][0]][dfs[-1][1]-2]==1:
            poss.append([dfs[-1][0], dfs[-1][1]-2])
        if dfs[-1][0]<2*height-2 and maze[dfs[-1][0]+2][dfs[-1][1]]==1:
            poss.append([dfs[-1][0]+2, dfs[-1][1]])
        if dfs[-1][1]<2*width-2 and maze[dfs[-1][0]][dfs[-1][1]+2]==1:
            poss.append([dfs[-1][0], dfs[-1][1]+2])
        if poss:
            choose = random.randint(0, len(poss)-1)
            maze[(poss[choose][0]+dfs[-1][0])//2][(poss[choose][1]+dfs[-1][1])//2] = 0
            dfs.append(poss[choose])
        else:
            dfs.pop()
    if mult_sols_allowed:
        for i in range(height):
            j = random.randint(1, height*2-2)
            k = random.randint(1, width*2-2)
            if any([maze[j][k-1]==0, maze[j][k+1]==0, maze[j-1][k]==0, maze[j+1][k]==0]):
                maze[j][k] = 0

# Using flood fill algorithm
def solve_maze(maze):
    maze_to_update = deepcopy(maze)
    curr = [start_point]
    maze_to_update[start_point[0]][start_point[1]]=1
    val = 1
    r = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    while maze_to_update[end_point[0]][end_point[1]]==0:
        new_curr = []
        val+=1
        while curr:
            t = curr.pop()
            for i, j in r:
                if maze_to_update[t[0]+i][t[1]+j]>val or maze_to_update[t[0]+i][t[1]+j]==0:
                    new_curr.append([t[0]+i,t[1]+j])
                    maze_to_update[t[0]+i][t[1]+j] = val
        curr = new_curr[:]
    curr_square = [end_point[0], end_point[1]]
    maze[curr_square[0]][curr_square[1]] = 1
    while curr_square!=start_point:
        for i, j in r:
            if maze_to_update[curr_square[0]+i][curr_square[1]+j]==maze_to_update[curr_square[0]][curr_square[1]]-1:
                curr_square = [curr_square[0]+i, curr_square[1]+j]
                break
        maze[curr_square[0]][curr_square[1]] = 1


def print_maze(maze):
    print(*["".join(["⬛" if j==-1 else "⬜" if j==0 else "🟩" for j in i]) for i in maze], sep = "\n")
    print("\n\n")

for i in range(2):start_point[i] = start_point[i]*2-1
for i in range(2):end_point[i] = end_point[i]*2-1
maze = []
for i in range(2*height+1):
    l = []
    for j in range(2*width+1):
        l.append(-1 if i%2==0 or j%2==0 else 1)
    maze.append(l)
create_maze(maze)

print("Unsolved Maze:")
print_maze(maze)
solve_maze(maze)
print("Solved Maze:")
print_maze(maze)
