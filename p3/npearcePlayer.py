"""
Name: npearcePlayer.py
Author: Nicholas Pearce
Date: 4/25/2018
About: "Intelligent" player of the game Atropos, utilizing iterative deepening 
and alpha-beta pruning for the minimax search tree.
Reference: Stuart Russell and Peter Norvig's Artificial Intelligence: A Modern 
Approach, specifically an algorithm which can be found at:
    http://aima.cs.berkeley.edu/python/games.html
"""
import sys
import string
import copy
import numpy as np

#msg = "Given board " + sys.argv[1] + "\n";
#sys.stderr.write(msg);

"""
For parsing board chunk of input string
Input: string containing values in each row of the board
Output: current board stored as 2D list (IGNORE RIGHT DISTANCE)
"""
def parseBoard(bstring):
    #split board string into row strings
    rows = string.split(bstring, "]")
    #get number of rows in board
    size = len(rows) - 1
    #initialize board structure filled with -1's
    b = [[-1 for x in range(size)] for y in range(size)]
    #store each value from input string in structure and return
    for row in rows:
        ldist = 0                       #reset left distance
        size -= 1                       #for indexing values
        if(size == 0):                  #skip first index if last row
            ldist += 1
        vals = string.lstrip(row, "[")  #strip leading bracket
        for val in vals:
            b[size][ldist] = int(val)
            ldist += 1
    return b
"""
For parsing last play chunk of input string
Input: string containing color and coordinates of previous play
Output: last play stored as tuple of length 3 (IGNORE RIGHT DISTANCE)
"""
def parsePlay(pstring):
    #return empty tuple if there is no last play
    if(pstring == "null"):
        return ()
    stripped = string.strip(pstring, "()")  #strip parentheses
    vals = string.split(stripped, ",")      #split into values
    color = int(vals[0])                    #convert color
    bottom = int(vals[1])                   #convert bottom distance
    left = int(vals[2])                     #convert left distance
    return (color, bottom, left)            #return tuple of distances

"""
For testing legality of individual triangles formed by a play
Input: three integers corresponding to values at positions in triangle
Output: boolean representing triangle's legality
"""
def isLegalTriangle(a, b, c):
    isOne = (a == 1 or b == 1 or c == 1)
    isTwo = (a == 2 or b == 2 or c == 2)
    isThree = (a == 3 or b == 3 or c == 3)
    return not(isOne and isTwo and isThree)

"""
For either returning colors of legal moves at a given position
Input: 2D list board B, tuple position pos, optional boolean isJump
Output: list of integer colors which can legally placed at pos
"""
def checkPosition(B, pos):
    #convert tuple to individual coordinates
    bdist = pos[0]
    ldist = pos[1]
    #list of tuples of coordinate adjustments to explore neighborhood of pos
    hood = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    #for storing legal moves
    moves = []
    #test each color
    for c in range(1, 4):
        allLegal = True
        #search through neighborhood two at a time
        for i in range(6):
            adjust1 = hood[i]
            adjust2 = hood[(i+1)%6]
            color1 = B[bdist + adjust1[0]][ldist + adjust1[1]]
            color2 = B[bdist + adjust2[0]][ldist + adjust2[1]]
            #skip unnecessary computations
            if(color1 != 0 and color2 != 0 and color1 != color2):
                l = isLegalTriangle(c, color1, color2)
                if not l:
                    allLegal = False
                    break
        if allLegal:
            moves.append(c)
    return moves

"""
For searching last play's neighborhood for legal moves
Input: 2D list board B, tuple position of last play lp
Output: list of tuples of legal moves
"""
def searchNeighborhood(B, lp):
    #enumerate tuple elements
    bdist = lp[1]
    ldist = lp[2]
    #list of tuples of coordinate adjustments to explore neighborhood of lp
    hood = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    #fill list with tuples of uncolored positions in neighborhood
    uncolored = []
    for adjust in hood:
        badjust = bdist + adjust[0]
        ladjust = ldist + adjust[1]
        if(B[badjust][ladjust] == 0):
            uncolored.append((badjust, ladjust))
    #return empty list if no uncolored positions in neighborhood
    if not uncolored:
        return uncolored
    #otherwise find legal colors at each uncolored position
    else:
        moves = []
        for pos in uncolored:
            colors = checkPosition(B, pos)
            for c in colors:
                moves.append((c,) + pos)
        #store losing move with signifier if player has lost
        if not moves:
            moves.append((-1, -1, -1))
            moves.append((1,) + uncolored[0])
        return moves

"""
For finding all legal moves in the case of a jump (or an initial move)
Input: 2D list board B
Output: list of tuples of legal moves
"""
def jumpSearch(B):
    #fill list with all uncolored positions on board
    positions = []
    for b in range(len(B)):
        for l in range(len(B[0])):
            if(B[b][l] == 0):
                positions.append((b, l))
    #compile all legal moves at every uncolored position
    moves = []
    for pos in positions:
        legal = checkPosition(B, pos)       #get legal colors at each position
        for c in legal:                     #concatenate colors and positions
            moves.append((c,) + pos)
    #return losing move with signifier if there are no legal moves
    if not moves:
        moves.append((-1, -1, -1))
        moves.append((1,) + positions[0])
    return moves

"""
For producing board resultant from a given play made on a given board
Input: 2D list board B, tuple play p
Output: 2D list board after play p has been made
"""
def makePlay(B, p):
    #enumerate tuple elements
    color = p[0]
    bdist = p[1]
    ldist = p[2]
    #make copy of B, make play on copy, return copy
    BB = copy.deepcopy(B)
    BB[bdist][ldist] = color
    return BB

"""
For adding back in the right distance to the tuple play
Input: integer length of one side of 2D board representation bsize,
        tuple play  p
Output: tuple play with right distance
"""
def addRight(bsize, p):
    #enumerate tuple elements
    bdist = p[1]
    ldist = p[2]
    #compute right distance
    rdist = bsize - bdist - ldist
    #concatenate with tuple play and return
    return (p + (rdist,))

"""
For finding the best next play given the current board and last play
Input: 2D list board B, tuple last play lp
Output: tuple next play
Implements a modified version of the algorithm at:
    http://aima.cs.berkeley.edu/python/games.html
    From Stuart Russell and Peter Norvig's 
    Artificial Intelligence: A Modern Approach (2009)
"""
def alphaBetaSearch(B, lp):
    #function definitions:
    #for maximizing player
    def max_value(Bmax, lpmax, alpha, beta, depth):
        maxmoves = searchNeighborhood(Bmax, lpmax)
        if not maxmoves:
            maxmoves = jumpSearch(Bmax)
        #severely penalize losses for maximizer
        if (maxmoves[0] == (-1, -1, -1)):
            return float("-inf")
        #return number of legal move as heuristic evaluator at terminal depth
        if(depth > 1):
            return len(maxmoves)
        v = float("-inf")
        for move in maxmoves:
            nextBoard = makePlay(Bmax, move)
            v = max(v, min_value(nextBoard, move, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    #for minimizing player
    def min_value(Bmin, lpmin, alpha, beta, depth):
        minmoves = searchNeighborhood(Bmin, lpmin)
        if not minmoves:
            minmoves = jumpSearch(Bmin)
        #strongly favor losses for minimizer
        if (minmoves[0] == (-1, -1, -1)):
            return float("inf")
        #return number of legal move as heuristic evaluator at terminal depth
        if(depth > 1):
            return -(len(minmoves))
        v = float("inf")
        for move in minmoves:
            nextBoard = makePlay(Bmin, move)
            v = min(v, max_value(nextBoard, move, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    #body of alphaBetaSearch:
    moves = []
    #only search neighborhood if there is a last move
    if(lp != ()):
        moves = searchNeighborhood(B, lp)
    #check if jump is needed
    if not moves:
        moves = jumpSearch(B)
    #return losing move if player has lost
    if (moves[0] == (-1, -1, -1)):
        return moves[1]
    #otherwise search recursively
    else:
        chances = []             #for storing probabilities of winning strategy
        for move in moves:
            nextBoard = makePlay(B, move)
            chances.append(min_value(nextBoard, move, float("-inf"), float("inf"), 0))
        idx = np.argmax(chances)
        return moves[idx]

#parse the input string
chunks = string.split(sys.argv[1], "LastPlay:")
board = parseBoard(chunks[0])
lastPlay = parsePlay(chunks[1])

#perform intelligent search to determine the next move
nextPlay = alphaBetaSearch(board, lastPlay)
nextPlayFinal = addRight(len(board), nextPlay)

#print to stdout for AtroposGame
sys.stdout.write(str(nextPlayFinal));
