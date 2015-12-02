# Community-Detection
Implement a community detection algorithm using a divisive hierarchical clustering (Girvan-Newman algorithm)

## Overview

Implement a community detection algorithm using a divisive hierarchical clustering (Girvan-Newman algorithm). It will make use of 2 python libraries called networkx and community. The networkx is a python library which can be installed on your machines. The assignment will require making use of the betweenness function and the modularity function which are a part of the networkx and the community libraries respectively. The matplotlib library will be used for plotting the communities.

## Main Steps

- Read the input file into a graph using Networkx.
- Use the betweenness of the edges as a measure to break communities into smaller
- communities in divisive clustering.
- The result should be the set of communities that have the highest modularity.
- Use the modularity function in the community API
- Use the betweenness function from networkx
- After the best set of communities are obtained, use matplotlib and networkx functions to plot the communities with different colors.

## Execution Details

The python code should take in two parameters, namely input file containing the graph and the output image with the community structure. For example:
Python communities.py input.txt image.png

## Input Parameters

### input.txt 
This file consists of the representation of the graph. All graphs tested will be undirected graphs. Each line in the input file is of the format:
1 2
where 1 and 2 are the nodes and each line represents an edge between the two nodes. The nodes are separated by one space.

### image.png
This will be a visualization of the communities detected by your algorithm. You should represent each communities in a unique color. Each node should contain a label representing the node numbers in the input file.

## Output 
The Python code should output the communities in the form of a dictionary to standard output (the console). Each community should be an array representing nodes in that community. In each array, the nodes should be sorted lexicographically. For example:

        [1,2,3,4] 
        [5,6,7,8] 
        [9]
        [10]

These 4 arrays represent the 4 communities.








