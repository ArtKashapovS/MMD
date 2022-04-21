import random
import math
import copy

## Local imports
from .atom import Atom

class Simulation:

    # Параметры для решения 
    kb = 1.380e-23 # Постояная Больцмана (Дж/K)
    Nav = 6.022e23 # Число Авагадро молекулы/моль
    m = (39.95/Nav)*(10**-3) # масса атома , кг
    e = kb*120 # Глубина пот. ямы, Дж
    sigma = 3.4e-10 # sigma в потенциале Ленарда Джонса, м
    rcut = 2.25*sigma # радиус отсечки, м
    rcutsq = rcut**2 # квадрат р. от.
    temp = 90 # Температура, K
    numAtoms = 864 # Число атомов для моделирования
    lbox = 10.229*sigma # Длина коробки, (м)
    dt = 1e-14 # шаг по времени, сек
    nSteps = 10 # Количество шагов
    currentTemp = 0

    atoms = []
    temperatures = [] # список для температур
    potentials = [] # Список для пот. энергий
    
    def __init__(self):
        print("Запуск..."),
        for i in range(0,self.numAtoms):
            self.atoms.append(Atom())
        self.assignPositions()
        self.applyBoltzmannDist()
        self.correctMomenta()
        print("Готово.")
        print("Идет вычисление.")
        
    def assignPositions(self):
        
        n = int(math.ceil(self.numAtoms**(1.0/3.0))) # Число атомов в одном направлении
        particle = 0
        
        for x in range(0, n):
            for y in range(0, n):
                for z in range(0, n):
                    if (particle < self.numAtoms):
                        self.atoms[particle].x = x * self.sigma
                        self.atoms[particle].y = y * self.sigma             
                        self.atoms[particle].z = z * self.sigma
                        particle += 1

    def applyBoltzmannDist(self):
        normDist = []
        scaling_factor = math.sqrt(self.kb*self.temp/self.m)

        # По распределению Максвелла
        for i in range(0, 3*self.numAtoms):
            normDist.append(random.gauss(0,1))
      
        # Нормировка
        for number in range(0, 3*self.numAtoms):
            normDist[number] = normDist[number]*scaling_factor
            
        # Распределение скоростей
        for atom in range(0, self.numAtoms):
            self.atoms[atom].vx = normDist[atom*3]
            self.atoms[atom].vy = normDist[atom*3+1]
            self.atoms[atom].vz = normDist[atom*3+2]

    def correctMomenta(self):
        sumvx = 0
        sumvy = 0
        sumvz = 0
        
        for atom in range(0, self.numAtoms):
            sumvx += self.atoms[atom].vx
            sumvy += self.atoms[atom].vy
            sumvz += self.atoms[atom].vz
        
        for atom in range(0, self.numAtoms):
            self.atoms[atom].vx -= sumvx/self.numAtoms
            self.atoms[atom].vy -= sumvy/self.numAtoms
            self.atoms[atom].vz -= sumvz/self.numAtoms
            
        
    def runSimulation(self, step, numSteps):
        self.updateForces()
        self.verletIntegration()
        self.updateTemperature()
        self.updatePotentials()
        self.resetForces()
        if (step+1) % 10 == 0:
            print("----------------- Прогресс: " + str(step+1) + "/" + str(numSteps) + " --------------------")
            print("Осталось (по моим расчетам, в минутах): " + str((1.7/60)*(numSteps-step)))

        if step > 20 and step < 120:
            self.scaleTemperature()
        
    def updateForces(self):
        for atom1 in range(0, self.numAtoms-1):
            for atom2 in range(atom1+1, self.numAtoms):
                self.calculateForce(atom1, atom2)
                    
        for atom in range(0, self.numAtoms):
            self.atoms[atom].fx *= 48*self.e
            self.atoms[atom].fy *= 48*self.e
            self.atoms[atom].fz *= 48*self.e
            self.atoms[atom].potential *= 4*self.e
            
    def calculateForce(self, atom1, atom2):
        
        # Расстояние между атомами
        dx = self.atoms[atom1].x - self.atoms[atom2].x
        dy = self.atoms[atom1].y - self.atoms[atom2].y
        dz = self.atoms[atom1].z - self.atoms[atom2].z
        

        dx -= self.lbox*round(dx/self.lbox)
        dy -= self.lbox*round(dy/self.lbox)
        dz -= self.lbox*round(dz/self.lbox)
        
        r2 = dx*dx + dy*dy + dz*dz

        if r2 < self.rcutsq:
            fr2 = (self.sigma**2)/r2
            fr6 = fr2**3
            force = fr6*(fr6 - 0.5)/r2
            pot = fr6*(fr6 - 1)
            
            # Update forces
            self.atoms[atom1].fx += force*dx
            self.atoms[atom2].fx -= force*dx
            self.atoms[atom1].fy += force*dy
            self.atoms[atom2].fy -= force*dy
            self.atoms[atom1].fz += force*dz
            self.atoms[atom2].fz -= force*dz
            
            # Update potentials
            self.atoms[atom1].potential += pot
            self.atoms[atom2].potential += pot
            
    def verletIntegration(self):
        # Интегрирование Верле
        for atom in range(0, self.numAtoms):
            
            # Обновляем скорости
            self.atoms[atom].vx += (self.atoms[atom].fx/self.m)*self.dt
            self.atoms[atom].vy += (self.atoms[atom].fy/self.m)*self.dt
            self.atoms[atom].vz += (self.atoms[atom].fz/self.m)*self.dt
            
            
            # Обновляем положения
            newX = self.atoms[atom].x + self.atoms[atom].vx*self.dt
            newY = self.atoms[atom].y + self.atoms[atom].vy*self.dt
            newZ = self.atoms[atom].z + self.atoms[atom].vz*self.dt

            # Обновляем текущие положения (пер. гр. усл.)
            if newX < 0:
                self.atoms[atom].x = newX + self.lbox
            elif newX > self.lbox:
                self.atoms[atom].x = newX - self.lbox
            else:
                self.atoms[atom].x = newX
            
            if newY < 0:
                self.atoms[atom].y = newY + self.lbox
            elif newY > self.lbox:
                self.atoms[atom].y = newY - self.lbox
            else:
                self.atoms[atom].y = newY
                
            if newZ < 0:
                self.atoms[atom].z = newZ + self.lbox
            elif newZ > self.lbox:
                self.atoms[atom].z = newZ - self.lbox
            else:
                self.atoms[atom].z = newZ

    def resetForces(self):
        for atom in range(0, self.numAtoms):
            self.atoms[atom].fx = 0
            self.atoms[atom].fy = 0
            self.atoms[atom].fz = 0
            self.atoms[atom].pot = 0
            
    def updateTemperature(self):
        # Вычисление текущей температуры системы
        sumv2 = 0
        for atom in self.atoms:
            sumv2 += atom.vx**2 + atom.vy**2 + atom.vz**2
        self.currentTemp = (self.m/(3*self.numAtoms*self.kb))*sumv2
        self.temperatures.append(self.currentTemp)
    
    def updatePotentials(self):
        # Вычисление текущей пот. энергии системы
        epot = 0
        for atom in self.atoms:
            epot += atom.potential
        self.potentials.append(epot)
    
    def getAtoms(self):
        return copy.deepcopy(self.atoms)
        
    def scaleTemperature(self):
        if self.currentTemp > 100.0 or self.currentTemp < 80.0:
            print("Обновляем скорости...")
            for atom in range(0, self.numAtoms):
                self.atoms[atom].vx *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vy *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vz *= math.sqrt(self.temp/self.currentTemp)
