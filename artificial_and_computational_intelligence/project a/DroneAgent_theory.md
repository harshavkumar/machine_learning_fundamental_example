
1. PEAS:
    <br>
    Defining minimal possible requirements our system should have to solve the problem in given setting*

    * Enviornment - The start point, destination, vegetation and other obstacles, radar interference as a 2D grid
    * Performamce measure - Number of cells it takes to get to destination undetected. (Better measure: Route length/ straight line dist. bw start and dest.)
    * Sensors - Satelite signal detectors, Radar detector, user signal detector, visual/sonic input to identify vegetation laid no fly zones
    * Actuators - Motor control to allow drone movement, turning and altitude change.

    Note : before the drone takes off the user already knows the best path from local beam search and is then supposed to decide risk and send
    the drone
-------------------------------------------------------------------------------------------------------------------------------------------
2. Data Structures:
    <br>
    Reference: https://www.youtube.com/watch?v=jhoXO1XF6Fk&ab_channel=GateSmashers

    * Output: Optimal cost AS WELL AS optimal path
    * Data Structutes: 
        * Priority Queue for open list,  
        * List for closed list
        * Array for grid: {'rowindex_colindex': (isStart, isGoal, isExplorable, Prob1, Prob2)}
        * Dictionary for path: {cell: parent_cell}

-------------------------------------------------------------------------------------------------------------------------------------------
3. Algorithm description - 

    Local beam intialises k inital states randomly - one of 5C3 possible state sequences is picked up randomly

    For each node, the drone can move in all 8 of its neighbours, except the red cells. Each time it traverses a cell we increment the cost by 1
    Basically cost of any path is number of cells the drone traversed to reach it. 
    This is under the assumption that drone moves freely in 2d space and crosses each cell with same speed, hence "earliest" is lowest cell count path

    Transition matrix from problem statement is created (cell with its evaluation function value)

    Local beam search with k=3:
    If goal is in intial states, we stop else continue

    For the first element of priority queue, we explore the cell and keep all of its reachable unclosed neighbours in open list
    We send this first element itself to closed list so it wont be explored in the future

    If any of the neigbours is in goal state we say we have arrived and stop the process. (Goal state eval fnc is taken arbitarily low)

    For all of its neighbours, if a path already exists then it must be the lowest evaluated, since we are greedily exploring.
    If the cost from current cell is no higher than the existing cells, we replace its parent with current cell.
    Basically we want to reduce cost wherever we can.

    Now in the current open list pick only k best elements and in the next iteration pick best element in priority queue

    We do this till either some neighbour of current node is the goal, or the open list is exhausted meaning the goal and start are disconnected components (It is possible due to existence of the red unreachable cells)

    In case the goal is unreachable from current set of inital states, we should ideally take another of the 5C3 combinations, however for our scenario a path always exists so we omit this step

-------------------------------------------------------------------------------------------------------------------------------------------

4. Space Time complexity

    Time complexity of local beam search: O(transition_matrix_edge^2 * (k * log k)):
    This is because in the worst case where goal is surrounded by boundaries and red cells on all side, our algorithm will explore all other nodes and return goal_flag False eventually. For each of these frontiers, we have to sort the entire open list that can have upto k+7 elements

    Space complexity of local beam search: O(transition_matrix_edge^2 +  k):
    path can contain keys as many as number of cells possible, apart from that we also keep closed list which also could go as long as number of cells possble. Apart from that we are also maintaing a transition matrix and an open list that goes atmax k+7 

--------------------------------------------------------------------------------------------------------------------

5. Hyperparameters

    k is the hyperparameter, along with seed in this case

    * k = 1 makes it hill climbing and k= inf makes it bfs, in general as k increases we can keep more nodes in memory and find better estimates locally
    * Since we have 5 inital states and k=3, we randomly choose any one out of possible 5C3 combinations. Different set of inital states might give a different optimal route cost and path.


