# Checkers

CSC384H1: Introduction to Artificial Intelligence\
Assignment 2

This program features an implementation of checkers, is able to generate all successors for black and white moves, and uses AlphaBeta Search to solve Checkers endgame puzzles.

## Introduction: Checkers

The version of Checkers implemented is the Standard English Draughts. 
In particulair, jumping is mandatory, and if a player has the option to jump, they must take it, even if doing so results in a disadvantage for the jumping player. 
After one jump, if the moved piece can jump another opponent's piece, it must keep jumping until no more jumps are possible, even if the jump is in a different diagonal direction.

## Input and Output File formats

We will represent each state in the following format.

- Each state is a grid of 64 characters. The grid has eight rows with eight characters per row.
- `.` (the period character) denotes an empty square.
- `r` denotes a red piece,
- `b` denotes a black piece,
- `R` denotes a red king,
- `B` denotes a black king.

Here is an example of a state:

```
........
....b...
.......R
..b.b...
...b...r
........
...r....
....B...
```

## Usage

`checkers.py` contains an implementation of AlphaBeta Search to solve checkers endgame puzzles.

To run this code, use the following commands:

```
python3 checkers.py --inputfile <input file> --outputfile <output file>
```

For example, to run AlphaBeta Search on the provided basic starting state `checkers_starting_state.txt` and output the path found by AlphaBeta Search in `checkers_sol.txt`, use the following command:

```
python3 checkers.py --inputfile checkers_starting_state.txt --outputfile checkers_sol.txt
```

Also provided is a sample endgame checkers state, `checkers_endgame.txt`. To run AlphaBeta Search on this endgame state, and output the path found by AlphaBeta Search in `checkers_endgame.txt`, use the following command:

```
python3 checkers.py --inputfile checkers_endgame.txt --outputfile checkers_endgame.txt
```
