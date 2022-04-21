import math
import numpy
import matplotlib.pyplot as plot

class Analysis:
    
    kb = 1.380e-23 # Постоная Больцмана (Дж/K)
    sigma = 3.4e-10 # sigma в потенциале Ленарда-Джонса, м
    dr = sigma/10

    V = (10.229*sigma)**3 # Объем коробки
    numAtoms = 864 # Число атомов
    dt = 1e-14
    
    originalAtoms = [] # Список для атомамов
    currentAtoms = []
    nr = [] # число частиц на расстоянии r
    velacfinit = 0 # авто кор. скоростей в момент времени t=0
    velacf = 0 # в след.
    lbox = 10.229*sigma # Длина коробки
    
    velaclist = [] # АКС для моделирование
    radiuslist = []
    timelist = []
    
    def __init__(self, atoms):
        self.originalAtoms = atoms
        
    def updateAtoms(self, atoms):
        self.currentAtoms = atoms
        
    def pairDistributionFunction(self):
        atom_counts = [0]*50
        cur_r = 0
        
        print("Создание радиального распределения..."),
        
        for atom1 in range(0, self.numAtoms-1):
            for atom2 in range(1, self.numAtoms):
                dx = self.currentAtoms[atom1].x - self.currentAtoms[atom2].x
                dy = self.currentAtoms[atom1].y - self.currentAtoms[atom2].y
                dz = self.currentAtoms[atom1].z - self.currentAtoms[atom2].z

                dx -= self.lbox*round(dx/self.lbox)
                dy -= self.lbox*round(dy/self.lbox)
                dz -= self.lbox*round(dz/self.lbox)
                
                r2 = dx*dx + dy*dy + dz*dz
                r = math.sqrt(r2)
                
                for radius in range(0, 50):
                    if (r < ((radius+1)*self.dr)) and (r > radius*self.dr):
                        atom_counts[radius] += 1
            
        for radius in range(1, 50):
            atom_counts[radius] *= (self.V/self.numAtoms**2)/(4*math.pi*((radius*self.dr)**2)*self.dr)
        print("Готово.")    
        return(atom_counts)        
                    
    def velocityAutocorrelation(self, step):
        vx = 0
        vy = 0
        vz = 0
        if step == 0:
            for atom in range(0, self.numAtoms):
                vx += self.originalAtoms[atom].vx * self.currentAtoms[atom].vx
                vy += self.originalAtoms[atom].vy * self.currentAtoms[atom].vy
                vz += self.originalAtoms[atom].vz * self.currentAtoms[atom].vz
            self.velacfinit += vx + vy + vz
            self.velacfinit /= self.numAtoms
            self.velaclist.append(self.velacfinit)
        else:   
            for atom in range(0, self.numAtoms):
                vx += self.originalAtoms[atom].vx * self.currentAtoms[atom].vx
                vy += self.originalAtoms[atom].vy * self.currentAtoms[atom].vy
                vz += self.originalAtoms[atom].vz * self.currentAtoms[atom].vz
            self.velacf += vx + vy + vz
            self.velacf /= self.numAtoms*self.velacfinit
            self.velaclist.append(self.velacf)
            self.velacf = 0
    
    def getVAC(self):
        return self.velaclist

    def plotRDF(self):
        pass
        rdf = numpy.loadtxt("rdf.csv")
        for radius in range(0,50):
            self.radiuslist.append(radius*self.dr)
        plot.figure()
        plot.plot(self.radiuslist, rdf)
        plot.xlabel('r')
        plot.ylabel('Радиальное распределение')
        plot.show()
        

    def plotVAC(self, nSteps):
       vac = numpy.loadtxt("vac.csv")
       vac[0] = 1
       for time in range(0, nSteps):
           self.timelist.append(float(time) * self.dt)
       plot.figure()
       plot.plot(self.timelist, vac)
       plot.xlabel('t')
       plot.ylabel('Авто корреляция скоростей')
       plot.grid()
       plot.show()

    def plotEnergy(self, temperatures, potentials, nSteps):

        KE = []
        for temp in temperatures:
            KE.append(3*self.numAtoms*self.kb*temp/2)
        
        steplist = []
        for time in range(0, nSteps):
            steplist.append(float(time))
        
        etot = []
        for energy in range(0, nSteps):
            etot.append(KE[energy] + potentials[energy])
            
        plot.figure()
        plot.plot(steplist, KE, label='Екин')
        plot.plot(steplist, potentials, label='Епот')
        plot.plot(steplist, etot, label='Еполн')
        plot.xlabel('Шаг')
        plot.ylabel('Энергия')
        plot.legend()
        plot.grid()
        plot.show()

    def plotTemperature(self, temperatures, nSteps):
    
        steplist = []
        for time in range(0, nSteps):
            steplist.append(float(time))
            
        plot.figure()
        plot.plot(steplist, temperatures)
        plot.xlabel('Шаг')
        plot.ylabel('Температура')
        plot.legend()
        plot.grid()
        plot.show()