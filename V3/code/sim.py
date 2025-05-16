import math 
import numpy as np
import random 
import os
import stat
WIDTH, HEIGHT = 600, 600

G = 6.67428e-11 


def write_array_to_file(array, filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
             for item in array:
                f.write("\n")
                f.write(str(item))
        os.chmod(filename, stat.S_IREAD) 
    else:
        print(f"File {filename} already exists. No action taken.")

class OrbitalObject:
    def __init__(self, x, y, radius, mass):
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.ship = False
        self.xv = 0
        self.yv = 0
        self._track_orbit = False
        self.orbit = []

    def enable_orbit_tracking(self, enable=True):

        self._track_orbit = enable
        if not enable:
            self.orbit = []  

    def graviation(self, other):

        distanceX = other.x - self.x
        distanceY = other.y - self.y
        

        dist_squared = distanceX**2 + distanceY**2
        distance = math.sqrt(dist_squared)
        

        if distance < 0.1:
            return 0, 0
        

        force_magnitude = G * self.mass * other.mass / dist_squared
        

        inv_distance = 1.0 / distance
        dir_x = distanceX * inv_distance
        dir_y = distanceY * inv_distance
        
        return force_magnitude * dir_x, force_magnitude * dir_y

    def update1(self, gravObjs):
        tForceX = 0
        tForceY = 0
        
        for obj in gravObjs:
            if self == obj:
                continue
            ForceX, ForceY = self.graviation(obj)
            tForceX += ForceX
            tForceY += ForceY

        self.xv += tForceX / self.mass
        self.yv += tForceY / self.mass
        
        self.x += self.xv
        self.y += self.yv
        
        if self._track_orbit:
            self.orbit.append((self.x, self.y))

    def calculateDistance(self, ziel):
        dx = self.x - ziel.x
        dy = self.y - ziel.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_contact(self, ziel):
        return self.calculateDistance(ziel) <= self.radius + ziel.radius
    
    def no(self, ziel, maxDistance, ID=""):
        """(0 to 1) to target"""
        var1 = self.calculateDistance(ziel)
        normVar1 = 1 - var1 / maxDistance
        return max(0, min(1, normVar1))  
        
    def normalizeDistanceShipPos(self, ziel, maxDistance):
        var1 = self.calculateDistance(ziel)
        normVar1 = 1 - var1 / maxDistance
        return max(0, min(1, normVar1)) 

class zielClass:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

def generate_random_orbital_position(planet_x, planet_y, exclude_x=None, exclude_y=None, 
                                   planet_min_distance=100, planet_max_distance=250,
                                   exclusion_distance=30,
                                   margin=20):
    """
    1. Distance from planet between min and max
    2. Not within exclusion zone around another objects
    3. In screen bounds
    """
    attempts = 0
    max_attempts = 5000  
    #RANDOM POSITION GENERATION
    while attempts < max_attempts:
        angle = random.uniform(0, 2 * math.pi)
        
        planet_distance = random.uniform(planet_min_distance, planet_max_distance)
        pos_x = planet_x + planet_distance * math.cos(angle)
        pos_y = planet_y + planet_distance * math.sin(angle)
        
        if (margin < pos_x < WIDTH - margin and 
            margin < pos_y < HEIGHT - margin):
            
            if exclude_x is not None and exclude_y is not None:
                exclude_distance = math.sqrt((pos_x - exclude_x)**2 + (pos_y - exclude_y)**2)
                if exclude_distance > exclusion_distance:
                    return pos_x, pos_y
            else:
                return pos_x, pos_y
        
        attempts += 1
    
    return random.uniform(margin, WIDTH - margin), random.uniform(margin, HEIGHT - margin)

planet1 = OrbitalObject(300, 300, 40, 10**13)
ship = OrbitalObject(50, 300, 5, 100)
ship.ship = True
ziel1 = zielClass(400, 200, 10)

# Global state
objs = [planet1, ship]
ziele = [ziel1]
cords = []

def restart(randomS, randomZ):
    global planet1, ship, ziel1, objs, ziele, cords
    
    cords = []
    
    planet1 = OrbitalObject(300, 300, 40, 10**13)
    planet1.xv = 0
    planet1.yv = 0

    
    if randomS:
        ship_x, ship_y = generate_random_orbital_position(
            planet1.x, planet1.y, 
            exclude_x=None, exclude_y=None
        )
        ship = OrbitalObject(ship_x, ship_y, 5, 100)
        ship.ship = True 
        
        dx = ship_x - planet1.x
        dy = ship_y - planet1.y
        distance = math.sqrt(dx**2 + dy**2)
        orbit_speed = math.sqrt((G * planet1.mass) / distance) * 0.8
        
        ship.xv = -dy / distance * orbit_speed
        ship.yv = dx / distance * orbit_speed
    else: 
        ship = OrbitalObject(50, 300, 5, 100)
        ship.ship = True    
        ship.xv = 0
        ship.yv = 1.2
    
    if randomZ:
        ziel_x, ziel_y = generate_random_orbital_position(
            planet1.x, planet1.y,
            exclude_x=ship.x, exclude_y=ship.y,
            planet_min_distance=100, planet_max_distance=200,  
            exclusion_distance=50 
        )
        ziel1 = zielClass(ziel_x, ziel_y, 10)
    else: 
        ziel1 = zielClass(400, 200, 10)

    objs = [planet1, ship]
    ziele = [ziel1]
    
    return next(5)[0]

def next(action):
    global cords
    
    terminated = False  
    positive = False
    fakefuel = 0

    for obj in objs:
        obj.update1(objs)

    if action == 0:
        fakefuel = -0.2
        ship.xv = ship.xv + 0.5
    elif action == 1:
        fakefuel = -0.2
        ship.xv = ship.xv - 0.5 
    elif action == 2:
        fakefuel = -0.2
        ship.yv = ship.yv + 0.5 
    elif action == 3:
        fakefuel = -0.2
        ship.yv = ship.yv - 0.5 
    elif action == 4: #no action
        fakefuel = 0
    elif action == 5:
        pass  # new start 
    
    cords.append((ship.x, ship.y))

    distanceZiel = ship.normalizeDistanceShipPos(ziel1, 150)
    
    State = [ship.x, ship.y, ship.xv, ship.yv, distanceZiel]
    assert len(State) == 5 #überprufung
    
    reward = 0
    rewardDistance = distanceZiel
  
    # Check for goal
    if ship.check_contact(ziel1):
        write_array_to_file(cords, "cords.txt")
        reward = 10
        positive = True
        terminated = True
        
        ship_x, ship_y = generate_random_orbital_position(
            planet1.x, planet1.y
        )
        ship.x = ship_x
        ship.y = ship_y
        
        dx = ship_x - planet1.x
        dy = ship_y - planet1.y
        distance = math.sqrt(dx**2 + dy**2)
        orbit_speed = math.sqrt((G * planet1.mass) / distance) * 0.8
        
        ship.xv = -dy / distance * orbit_speed
        ship.yv = dx / distance * orbit_speed
        
        ziel_x, ziel_y = generate_random_orbital_position(
            planet1.x, planet1.y,
            exclude_x=ship.x, exclude_y=ship.y,
            planet_min_distance=100, planet_max_distance=200, 
            exclusion_distance=50
        )
        ziel1.x = ziel_x
        ziel1.y = ziel_y
        
        return np.array(State, dtype=np.float32), reward, terminated, positive

    elif distanceZiel > 0:
        reward = rewardDistance + fakefuel
    else:
        reward = fakefuel

    if ship.x < -200 or ship.x > (WIDTH+200) or ship.y < -200 or ship.y > (HEIGHT+200): 
        terminated = True
        positive = False
        
        ship_x, ship_y = generate_random_orbital_position(
            planet1.x, planet1.y
        )
        ship.x = ship_x
        ship.y = ship_y
    
        dx = ship_x - planet1.x
        dy = ship_y - planet1.y
        distance = math.sqrt(dx**2 + dy**2)
        orbit_speed = math.sqrt((G * planet1.mass) / distance) * 0.8
        
        ship.xv = -dy / distance * orbit_speed
        ship.yv = dx / distance * orbit_speed
        
        ziel_x, ziel_y = generate_random_orbital_position(
            planet1.x, planet1.y,
            exclude_x=ship.x, exclude_y=ship.y,
            planet_min_distance=100, planet_max_distance=200, 
            exclusion_distance=50
        )
        ziel1.x = ziel_x
        ziel1.y = ziel_y
        
        reward = -10
        return np.array(State, dtype=np.float32), reward, terminated, positive 

    return np.array(State, dtype=np.float32), reward, terminated, positive