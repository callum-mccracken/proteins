import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os

seq_str = "PHPPPHPHPPPH"
#seq_str = "PHPPPH"

# ensure this only contians Ps and Hs!
sequence = np.array(list(seq_str))

# create a dir for images to be stored, directory PHPPPH or whatever
if not os.path.exists(seq_str):
    os.mkdir(seq_str)

n_monomers = len(sequence)

# 0 for H, 1 for P
binary_sequence = np.zeros(n_monomers)
binary_sequence[sequence=="P"] = 1

def check_good(move_list):
    # move_list = [[0,1], ...] list of moves on a grid.

    # number of times we've reached each point on the grid
    grid = np.zeros((n_monomers*2, n_monomers*2), dtype=int)

    # start at the middle
    x = n_monomers
    y = n_monomers
    grid[x,y] = 1

    points = [[x,y]]
    for dx, dy in move_list:
        x += dx
        y += dy
        if grid[x,y] == 1:
            # self-intersection!
            return False, points
        grid[x,y] += 1
        points.append([x,y])

    # if we haven't returned, no self-intersections yet!

    # still have to check compactness though
    pts_reached = np.where(grid==1)
    min_x = min(pts_reached[0])
    max_x = max(pts_reached[0])
    min_y = min(pts_reached[1])
    max_y = max(pts_reached[1])
    # make compact array
    compact = np.zeros_like(grid)
    compact[min_x:max_x+1, min_y:max_y+1] = 1
    if not np.array_equal(compact, grid):  # not compact
        return False, grid
    return True, points

def plot(move_list, energy_penalty=None, n=None, show=False):
    plt.cla()
    plt.clf()
    plt.axis('equal')

    gray = "#CACACA"
    red = "#C46B6B"
    markersize= 40 if len(seq_str) == 6 else 20

    clist = [red if b == 0 else gray for b in binary_sequence]
    
    # start at the origin
    x = 0
    y = 0
    # color = green for the first dot, just for clarity
    plt.plot([x], [y], color=clist[0], marker='o', markersize=markersize, clip_on=False)
    for i, (dx, dy) in enumerate(move_list):
        plt.plot([x,x+dx], [y,y+dy], 'gray', linewidth=5, zorder=-1, clip_on=False)
        x += dx
        y += dy
        plt.plot([x], [y], color=clist[i+1], marker='o', markersize=markersize, clip_on=False)
    if energy_penalty is not None:
        plt.title(f"$\\varepsilon={energy_penalty}$", fontsize=20)

    plt.axis('off')
    plt.savefig(f'{seq_str}/plot{n}.png', bbox_inches='tight')
    if show:
        plt.show()


possible_moves = [[0,1], [0,-1], [1,0], [-1,0]]
n_moves = n_monomers-1

print("finding all possible conformations")
points_arr = []
moves_arr = []
for moves in tqdm([m for m in itertools.product(possible_moves, repeat=n_moves-1)]):
    # make sure the first move is [0,1] (up)
    moves = [[0,1]] + list(moves)
    # ensure first turn is right!
    if [1,0] in moves:
        r_index = moves.index([1,0])
        if [-1,0] in moves:
            l_index = moves.index([-1,0])
            if r_index > l_index:
                continue  # this ensures the first turn is right
    else:
        continue

    is_good, points = check_good(moves)
    if is_good:  # if compact and no self-intersections
        points_arr.append(points)
        moves_arr.append(moves)

print("found", len(points_arr))

def energy_penalty(type1, type2):
    # recall 0 for H, 1 for P
    if type1 == 0 and type2 == 0:
        # H--H has energy penalty 0
        return 0
    elif type1 == 0 and type2 == 1:
        # H--P has energy penalty 1
        return 1
    elif type1 == 1 and type2 == 0:
        # P--H has energy penalty 1
        return 1
    elif type1 == 1 and type2 == 1:
        # P--P has energy penalty 0
        return 0
    else:
        raise ValueError("why are you here?")


def total_energy_penalty(points, types):
    # types[i] = 0 or 1, for H or P respectively
    connections = []
    penalty = 0
    for i, point in enumerate(points):
        if types[i] == 0:  # only consider H monomers
            point_x, point_y = point
            up = [point_x, point_y+1]
            down = [point_x, point_y-1]
            right = [point_x+1, point_y]
            left = [point_x-1, point_y]
            # look at adjacent molecules
            for adj in [up, down, left, right]:
                if adj in points:
                    # there is a monomer adjacent to the point
                    if [point, adj] not in connections:
                        # and make sure it's not the previous or subsequent monomer:
                        if i == 0:
                            condition = adj != points[i+1]
                        elif i == len(points)-1:
                            condition = adj != points[i-1]
                        else:
                            condition = (adj != points[i-1]) and (adj != points[i+1])
                        if condition:
                            # if we haven't got this connection between point and adjacent
                            j = points.index(adj)
                            penalty += energy_penalty(types[i],types[j])
                            connections.append([point, adj])
                            connections.append([adj, point])
                else:
                    # connection between monomer and water
                    penalty += 1 if types[i] == 0 else 0
    return penalty

print("calculating energy penalties")
eps = np.zeros(len(points_arr))
for i, pts in tqdm(enumerate(points_arr)):
    eps[i] = total_energy_penalty(pts, binary_sequence)

for i in range(len(moves_arr)):
    #show = (eps[i] == min(eps))
    plot(moves_arr[i], eps[i], i)#, show=show)


print("min energy penalty:", min(eps))
print("degeneracy:", len(np.where(eps==min(eps))[0]))

plt.cla()
plt.clf()
plt.hist(eps)
plt.show()