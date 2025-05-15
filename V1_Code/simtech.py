import pygame
import math 
import numpy as np

import numpy as np  


WIDTH, HEIGHT = 600, 600
window = pygame.display.set_mode((WIDTH, HEIGHT),pygame.DOUBLEBUF,display=0)

WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)

#Heatmap Konfiguration (GRID):
numRows = 25
numCols = 25 
blockSize = 10
blockSpacing = 5
chungusbingus = 0 
# def write_array_to_file(array, filename):
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#              for item in array:
#                 f.write("\n")
#                 f.write(str(item))
#         os.chmod(filename, stat.S_IREAD)  # Change file permissions to read-only

#     else:
#         print(f"File {filename} already exists. No action taken.")



class OrbitalObject:
    G = 6.67428e-11# Gravitational constant


    def __init__(self, x, y, radius, color, mass):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color 
        self.mass = mass

        self.orbit = []
        
        self.ship = False #Ship ist was von Agent Kontrolliert wird. 
        
        self.xv = 0
        self.yv = 0

    def draw(self, window):
        x = self.x 
        y = self.y 


        # Orbit Zeichnen 
        if len(self.orbit) > 2: 
            
            updatedPoints = []
            for point in self.orbit:
                x, y = point 
                x = x 
                y = y 
                updatedPoints.append((x, y))

            pygame.draw.lines(window, self.color, False, updatedPoints, 2)
        pygame.draw.circle(window, self.color, (x, y), self.radius)


        if self.ship:
            xVector = self.xv * 10 #variations versuchen 
            yVector = self.yv *10

            pygame.draw.rect(window, BLUE, pygame.Rect(x, y, 10, yVector)) if yVector > 0 else pygame.draw.rect(window, BLUE, pygame.Rect(x, y + yVector, 10, -yVector))
            pygame.draw.rect(window, RED, pygame.Rect(x, y, xVector, 10)) if xVector > 0 else pygame.draw.rect(window, RED, pygame.Rect(x + xVector, y, -xVector, 10))


    def graviation(self, other):
        otherX, otherY = other.x, other.y
        distanceX = otherX - self.x
        distanceY = otherY - self.y
        
        distance = math.sqrt(distanceX**2 + distanceY**2) #Pythagoras
        
        force = 6.67428e-11 * self.mass * other.mass / distance**2 # F = G*m1*m2/r^2
        theta = math.atan2(distanceY, distanceX) # Winkel zwischen den beiden Objekten; atan2 ist gengen tans 
                                                                #     /|θ 
                                                                #    / |
                                                                #   /  |
                                                                #  /   |
                                                                # /____|
                                                                #       
        forceX = math.cos(theta) * force
        forceY = math.sin(theta) * force
        return forceX, forceY

    def update1(self, gravObjs):
        tForceX = 0
        tForceY = 0
        for obj in gravObjs:
            if self == obj:
                continue

            ForceX, ForceY = self.graviation(obj)
            tForceX += ForceX
            tForceY += ForceY


        self.xv += tForceX / self.mass * 1 #1 oder 3600 * 24
        self.yv += tForceY / self.mass * 1
        self.x += self.xv * 1
        self.y += self.yv * 1
        self.orbit.append((self.x, self.y))


    
    
    def calculateDistance( self, ziel):
        distance = ((self.x - ziel.x)**2 + (self.y - ziel.y)**2)**0.5
        return distance
    
    def check_contact(self, ziel):
        return self.calculateDistance(ziel) <= self.radius + ziel.radius # wenn die Distanz kleiner ist als die Summe der Radien dann ist der Kontakt erreicht
    
    def normalizeDistanceShipPos(self, ziel, maxDistance, ID):
        var1 = self.calculateDistance(ziel)
        normVar1 = 1 - var1 / maxDistance
        normVar1 = min(1, max(0, normVar1))
        #print(f'{ID} NDSP:{round(normVar1, 1)}')
        return round(normVar1, 1)



class zielClass:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.blocks = []

    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x, self.y), self.radius)
    
    def calculate_block_positions(self):    
        i1 = 0
        j1 = 0 
        blocksarr = []

        obj_center_x, obj_center_y = self.x, self.y
        half_grid_width = (numCols * blockSize + (numCols - 1) * blockSpacing) / 2
        half_grid_height = (numRows * blockSize + (numRows - 1) * blockSpacing) / 2

        for row in range(numRows):
            for col in range(numCols):
                block_x = obj_center_x - half_grid_width + col * (blockSize + blockSpacing)
                block_y = obj_center_y - half_grid_height + row * (blockSize + blockSpacing)
                blocksarr.append((block_x, block_y))
        
        return blocksarr
    
    def calculateDistancePos(self):
        print("wait")

    def drawBlocks(self):
        blocks = self.calculate_block_positions()
        for block in blocks:
            pygame.draw.rect(window, (255, 255, 255), (block[0], block[1], blockSize, blockSize))
   

run = True
varBlocks = False
  
planet1 = OrbitalObject(300,300, 40, BLUE, 10**13)


ship = OrbitalObject(50,300, 5, WHITE, 100)
ship.ship = True 

ship.xv = 0
ship.yv = 1.2

ziel1 = zielClass(400, 200, 10, RED)
ziel2 = zielClass(300, 300, 10, WHITE)

objs = [planet1,  ship]
ziele = [ziel1]

cords= []

def restart():
    cords = []
    #varBlocks = False
    
    planet1 = OrbitalObject(300,300, 40, BLUE, 10**13)
    planet1.xv = 0
    planet1.yv = 0

    ship = OrbitalObject(50,300, 5, WHITE, 100)
    ship.ship = True 
    ship.x = 50
    ship.y = 300    
    ship.xv = 0
    ship.yv = 1.2

    ziel1 = zielClass(400, 200, 10, RED)
    #ziel2 = zielClass(300, 300, 10, WHITE)

    objs = [planet1,  ship]
    ziele = [ziel1]
    return next(4)[0]
    # 'if sim: 
    #     while run:
    #         clock.tick(60)
    #         window.fill((0,0,0))
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 run = False
    #             if (ziel1.x == ship.x) and (ziel1.y and ship.y):
    #                 run = False
    #             if event.type == pygame.KEYDOWN:  
    #                 if event.key == pygame.K_a:  
    #                     ship.xv = ship.xv - 0.5 
    #                     print("- 0.5 m/s in x direction") 
    #                 if event.key == pygame.K_d:  
    #                     ship.xv = ship.xv + 0.5
    #                     print("+ 0.5 m/s in x direction") 
    #                 if event.key == pygame.K_w:  
    #                     ship.yv = ship.yv - 0.5
    #                     print("- 0.5 m/s in y direction") 
    #                 if event.key == pygame.K_s:  
    #                     ship.yv = ship.yv + 0.5
    #                     print("+ 0.5 m/s in y direction") 
    #                 if event.key == pygame.K_1:
    #                     varBlocks = not varBlocks

    #         for obj in objs:
    #             obj.update1(objs)
    #             obj.draw(window)

    #         for ziel in ziele:
    #             if ship.check_contact(ziel):
    #                 print("Ziel erreicht")
    #                 run = False
            
    #         ship.normalizeDistanceShipPos(ziel1, 250, "ziel1")
    #         ship.normalizeDistanceShipPos(planet1, 300, "planet1")

            
    #         ziel.draw(window)
    #         if varBlocks:
    #             ziel1.drawBlocks()
    #             ziel2.drawBlocks()
            





    #         pygame.display.update()
    #     sim = False
    #     pygame.quit()
        
has_written_to_file = False

cords = []

def next(action):
    #print("next")
    terminated = False  
    positive = False

    for obj in objs:
        obj.update1(objs)
        #obj.draw(window)

    if action == 0:
        ship.xv = ship.xv + 0.5
        #print("0.5 m/s in x direction")
    if action == 1:
        ship.xv = ship.xv - 0.5 
        #print("- 0.5 m/s in x direction")   
    if action == 2:
        ship.yv = ship.yv + 0.5 
        #print("0.5 m/s in y direction")
    if action == 3:
        ship.yv = ship.yv - 0.5 
        #print("- 0.5 m/s in y direction")
    # if action == 4:
    #    # print("no action")
       

    cords.append((ship.x, ship.y))

    distanceZiel = ship.normalizeDistanceShipPos(ziel1, 150, "ziel1")
    #ship.normalizeDistanceShipPos(planet1, 300, "planet1")

    # print("--------")
    # print("ship.xv: " + str(ship.xv) + " ship.x: " + str(ship.x))    
    # print("ship.yv: " + str(ship.yv) + " ship.y: " + str(ship.y)) 
    

    State = [ship.x, ship.y, ship.xv, ship.yv, distanceZiel] ### Später dinge wie gravitationskraft F oder Treibstoff hinzufügen
    assert len(State) == 5
    reward = 0
    rewardDistance = distanceZiel * 10 #0.5 *100 = 50 
  
    




    


    if ship.check_contact(ziel1):
       # write_array_to_file(cords, "cords3.txt")

        positive = True
        reward = 100
        terminated = True
        ship.xv = 0
        ship.x = 50
        ship.y = 300
        ship.yv = 1.2
        return np.array(State, dtype=np.float32), reward, terminated, positive



    elif distanceZiel > 0 :
        reward = 0
        reward = rewardDistance  #0.5 *10 = 5

    else:
        reward = 0


    if State[0] < -200 or State[0] > (WIDTH+200) or State[1] < -200 or State[1] > (HEIGHT+200):
        terminated = True
        positive = False
        ship.xv = 0
        ship.x = 50
        ship.y = 300
        ship.yv = 1.2
        reward = -100
        return np.array(State, dtype=np.float32), reward, terminated, positive 





    # print("^^^^^^^^")
    # print(distanceZiel)
    # print("reward")
    # print(reward)
    # print("locationss")
    # print(ship.x)
    # print(ship.y)
    # print("^^^^^^^^")
    return np.array(State, dtype=np.float32), reward, terminated, positive 

