import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def perimetre(pt1, pt2, pt3, pt4):
    p = math.dist(pt1, pt2) + math.dist(pt2, pt3) + \
        math.dist(pt3, pt4) + math.dist(pt4, pt1)
    return p

def aire(pt1, pt2, pt3, pt4):
    x = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y = [pt1[1], pt2[1], pt3[1], pt4[1]]
    a = 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0] - 
                 (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0]))
    return a

if len(sys.argv) < 2:
    print("Donner le nombre de carrés à calculer")
    sys.exit(1)

img = plt.imread('challenge1.png')
nb_carres = int(sys.argv[1])
resultats = []

for i in range(nb_carres):
    plt.imshow(img)
    plt.title(f"Carré {i+1}")
    points = plt.ginput(4)
    plt.close()
    
    p = perimetre(points[0], points[1], points[2], points[3])
    s = aire(points[0], points[1], points[2], points[3])
    
    resultats.append({
        'id': i + 1,
        'perimetre': p,
        'surface': s
    })

resultats.sort(key=lambda x: x['surface'], reverse=True)

surface_max = resultats[0]['surface']

print("\nClassement par surface décroissante :")
for res in resultats:
    part_relative = res['surface'] / surface_max
    print(f"Carré {res['id']} : Surface = {res['surface']:.2f} (Part = {part_relative:.2f}), Périmètre = {res['perimetre']:.2f}")