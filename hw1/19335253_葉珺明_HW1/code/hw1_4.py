import numpy as np
import random
SIZE = 7

def bool_in_grid(x, y):
    if x<0 or x>SIZE-1 or y<0 or y>SIZE-1:
        return 0
    else:
        return 1

def bool_not_visted(visited, x, y):
    if grid_visited[x,y]==2:
        return 0
    elif x==int(SIZE/2) and y==int(SIZE/2) and (visited[x,y]==0 or visited[x,y]==1):
        return 1
    else:
        return 1

def orientation_exsist(visited, x, y):
    path = []
    if bool_in_grid(x, y-1) and bool_not_visted(visited, x, y-1):
        path.append((x, y-1))
    
    if bool_in_grid(x, y+1) and bool_not_visted(visited, x, y+1):
        path.append((x, y+1))

    if bool_in_grid(x-1, y) and bool_not_visted(visited, x-1, y):
        path.append((x-1, y))

    if bool_in_grid(x+1, y) and bool_not_visted(visited, x+1, y):
        path.append((x+1, y))
    
    return path

def SEEK_PATH(route, visited, x, y):
    route.append((x,y))
    # print(x, y)
    if x==SIZE-1 and y==SIZE-1:
        # print(x, y)
        return route

    path = orientation_exsist(visited, x, y)
    if len(path)==0:
        return []
    # print(path)

    x, y = random.choice(path)
    if x==int(SIZE/2) and y==int(SIZE/2):
        visited[x, y] += 1
    else:
        visited[x, y] += 2

    return SEEK_PATH(route, visited, x, y)

sum = 0.0
for i in range (20000):
    grid_visited = np.zeros((SIZE, SIZE))
    grid_visited[0, 0] = 2
    path = []
    path = SEEK_PATH(path, grid_visited, 0, 0)
    if len(path):
        sum += 1
        # print(i, ':', path, '----------------------------')

print('Pro: ', sum/20000)






