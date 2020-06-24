"""
proteins.py

By Team 9 for the PHYS 380 Midterm, June 23, 2020.

This is a python module for calculating the ground state of an
arbitrary polymer using the HP model.

It's not super speedy, but it only needs to be able to
calculate up to a 12-mer, so oh well.
"""

# some standard inputs
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os

# put your favourite protein sequence here
# make sure to only include Ps and Hs in the string
#seq_str = "PHPPPHPHPPPH"
seq_str = "PHPPPH"

############# Initial Setup ###############

# number of monomers
n_monomers = len(seq_str)

# convert to numpy array
sequence = np.array(list(seq_str))

# convert to binary array, 0 for H, 1 for P
binary_sequence = np.zeros(n_monomers)
binary_sequence[sequence=="P"] = 1

# create directory for images
if not os.path.exists(seq_str):
    os.mkdir(seq_str)


########### Helper functions ################

def check_good(move_list):
    """
    Given a move list, we can think of the chain formed by
    starting at [0,0] and applying each move.

    This function answers the question "is the chain that
    is formed compact and non-self-intersecting?"
    (That's what is meant by "good" in check_good)

    move_list:
        list containing [0,1], [0,-1], [1, 0], [-1, 0]
        to denote the up, down, right, left moves for a
        polymer chain

    returns (bool, list):
        - bool = True if "good", False if not
        - list = list of points in the chain
    """
    # number of times we've reached each point on the grid
    grid = np.zeros((n_monomers*2, n_monomers*2), dtype=int)

    # start at the middle
    x = n_monomers
    y = n_monomers
    grid[x,y] = 1

    # start at the first point, then progress through moves
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
    if not np.array_equal(compact, grid):  # if not compact
        return False, points
    return True, points  # if compact (and no self-ints)

def turn_sequence(move_list):
    """
    Given a move list, we can think of the chain formed by
    starting at [0,0] and applying each move.

    Then we can think of the intersections along the path
    of the chain. At each intersection we either go straight (s),
    turn left (l), or turn right (r).

    This returns the string "srlr...", one character for each turn.

    move_list:
        list containing [0,1], [0,-1], [1, 0], [-1, 0]
        to denote the up, down, right, left moves for a
        polymer chain

    returns string:
        "s" "r" "l" for each turn
    """
    u = [0,1]
    d = [0,-1]
    r = [1, 0]
    l = [-1, 0]
    turn_seq = ""
    for i, move in enumerate(move_list[:-1]):
        next_move = move_list[i+1]
        if move == next_move:
            turn_seq += "s"  # s for straight
        elif move == u and next_move == r:  # turned right
            turn_seq += "r"
        elif move == u and next_move == l:  # turned left
            turn_seq += "l"
        elif move == d and next_move == r:
            turn_seq += "l"
        elif move == d and next_move == l:
            turn_seq += "r"
        elif move == r and next_move == u:
            turn_seq += "l"
        elif move == r and next_move == d:
            turn_seq += "r"
        elif move == l and next_move == u:
            turn_seq += "r"
        elif move == l and next_move == d:
            turn_seq += "l"
    return turn_seq

def plot(move_list, energy_penalty=None, n=None, show=False):
    """
    Given a move list, we can think of the chain formed by
    starting at [0,0] and applying each move.

    This makes a plot of the polymer chain.

    move_list:
        list containing [0,1], [0,-1], [1, 0], [-1, 0]
        to denote the up, down, right, left moves for a
        polymer chain
    energy_penalty:
        optional number to put a title on the graph
    n:
        optional integer for graph save file naming
    show:
        bool, whether or not to show the plot interactively
    """

    # basic graph config
    plt.cla()
    plt.clf()
    plt.axis('equal')

    # make 'er pretty
    gray = "#CACACA"
    red = "#C46B6B"
    markersize= 40 if len(seq_str) == 6 else 20

    # list of colours
    clist = [red if b == 0 else gray for b in binary_sequence]

    # start at the origin
    x = 0
    y = 0
    plt.plot([x], [y], color=clist[0], marker='o', markersize=markersize, clip_on=False)
    # progress through each point, plotting lines and points
    for i, (dx, dy) in enumerate(move_list):
        plt.plot([x,x+dx], [y,y+dy], 'gray', linewidth=5, zorder=-1, clip_on=False)
        x += dx
        y += dy
        plt.plot([x], [y], color=clist[i+1], marker='o', markersize=markersize, clip_on=False)

    # give title
    if energy_penalty is not None:
        plt.title(f"$\\varepsilon={energy_penalty}$", fontsize=20)
    # save, show if needed
    plt.axis('off')
    plt.savefig(f'{seq_str}/plot{n}.png', bbox_inches='tight')
    if show:
        plt.show()

def energy_penalty(type1, type2):
    """
    Return energy penalty for a single connection
    between monomers of type type1 and type2
    
    type1, type2:
        integers, either 0 or 1

    returns:
        integer, either 0 or 1
    """
    # recall type = 0 for H, 1 for P
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
    """
    For a chain of points of type H or P, what is the
    total energy penalty for the polymer formed by the points?

    points:
        list, each element is a [x,y] location of a monomer
    types:
        array, types[i] = 0 or 1, for H or P respectively

    returns:
        integer, energy penalty
    """
    # keep track of each connection between monomers (or water)
    connections = []

    # total penalty
    penalty = 0

    for i, point in enumerate(points):
        if types[i] == 0:  # only consider H monomers
            point_x, point_y = point
            # adjacent spots we could connect with
            up = [point_x, point_y+1]
            down = [point_x, point_y-1]
            right = [point_x+1, point_y]
            left = [point_x-1, point_y]
            # look at adjacent spots
            for adj in [up, down, left, right]:
                # if there is a monomer adjacent to the point
                if adj in points:
                    # and we haven't already had a connection to it
                    if [point, adj] not in connections:
                        # make sure it's not the previous or subsequent monomer:
                        if i == 0:
                            condition = adj != points[i+1]
                        elif i == len(points)-1:
                            condition = adj != points[i-1]
                        else:
                            condition = (adj != points[i-1]) and (adj != points[i+1])
                        if condition:
                            # if we haven't already seen this connection, add it to conections
                            j = points.index(adj)
                            penalty += energy_penalty(types[i],types[j])
                            connections.append([point, adj])
                            connections.append([adj, point])
                else:
                    # connection between monomer and water if needed
                    penalty += 1 if types[i] == 0 else 0
    return penalty


################# Main Code ##################
print("finding all possible conformations")


points_arr = []
moves_arr = []
unique_paths = []
possible_moves = [[0,1], [0,-1], [1,0], [-1,0]]
n_moves = n_monomers-1

# for each possible chain of moves
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
        continue  # this ensures we have at least one turn

    is_good, points = check_good(moves)
    if is_good:  # if compact and no self-intersections
        # get turn sequence
        turn_seq = turn_sequence(moves)
        # and check it's not already in the unique paths list:
        if turn_seq not in unique_paths:
            points_arr.append(points)
            moves_arr.append(moves)
            # add it and the reversed version so we don't get flipped verison too
            unique_paths.append(turn_seq)
            unique_paths.append(turn_seq[::-1])
print("found", len(points_arr))

print("calculating energy penalties")
eps = np.zeros(len(points_arr))
for i, pts in tqdm(enumerate(points_arr)):
    eps[i] = total_energy_penalty(pts, binary_sequence)
for i in range(len(moves_arr)):
    plot(moves_arr[i], eps[i], i)
print("min energy penalty:", min(eps))
print("degeneracy:", len(np.where(eps==min(eps))[0]))

plt.cla()
plt.clf()
plt.hist(eps)
plt.hist("Histogram of $\\varepsilon$ Values")
plt.savefig(f'{seq_str}/hist.png')
