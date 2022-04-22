import random, math, copy, os
import numpy as np, matplotlib.pyplot as plt


class Constants:

    def __init__(self) -> None:
        self.kb = 1.380e-23 # Постоная Больцмана (Дж/K)
        self.Nav = 6.022e23 # Число Авагадро молекулы/моль
        self.sigma = 3.4e-10 # sigma в потенциале Ленарда-Джонса, м
        self.dr = self.sigma / 10

        self.m = (39.95 / self.Nav) * (10**-3) # масса атома , кг
        self.e = self.kb * 120 # Глубина пот. ямы, Дж

        self.V = (10.229 * self.sigma)**3 # Объем коробки
        self.numAtoms = 864 # Число атомов
        self.dt = 1e-14 # шаг по времени, сек
        self.lbox = 10.229 * self.sigma #* math.pow(2.0, 1/3)# Длина коробки
        
        self.temp = 90 # Температура, K


class Atom:
    
    def __init__(self):
        # Положение
        self.x = 0
        self.y = 0
        self.z = 0
        
        # Скорость
        self.vx = 0
        self.vy = 0
        self.vz = 0
        
        # Сила
        self.fx = 0
        self.fy = 0
        self.fz = 0

        self.potential = 0;


class Simulation:

    # Параметры для решения
    consts = Constants() 
    kb = consts.kb
    Nav = consts.Nav
    m = consts.m
    e = consts.e
    sigma = consts.sigma
    temp = consts.temp
    numAtoms = consts.numAtoms
    lbox = consts.lbox
    dt = consts.dt

    Volm = consts.V

    rcut = 2.25 * sigma # радиус отсечки, м
    rcutsq = rcut**2 # квадрат р. от.

    currentTemp = 0

    atoms = []
    temperatures = [] # список для температур
    potentials = [] # Список для пот. энергий

    p_int = 0
    
    def __init__(self):
        print("Инициализация атомов..."),
        for i in range(0,self.numAtoms):
            self.atoms.append(Atom())
        self.assign_positions()
        self.apply_maxwell_dist()
        self.correct_mom()
        print("Готово.")
        print("Запуск вычислений...")
         
    def assign_positions(self):
        
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

    def apply_maxwell_dist(self):
        normDist = []
        scaling_factor = math.sqrt(self.kb * self.temp / self.m)

        # По распределению Максвелла
        for i in range(0, 3*self.numAtoms):
            normDist.append(random.gauss(0,1))
      
        # Нормировка
        for number in range(0, 3*self.numAtoms):
            normDist[number] = normDist[number]*scaling_factor

        abs_vel = []
        # Распределение скоростей
        for atom in range(0, self.numAtoms):
            self.atoms[atom].vx = normDist[atom*3]
            self.atoms[atom].vy = normDist[atom*3+1]
            self.atoms[atom].vz = normDist[atom*3+2]
            abs_vel.append(np.sqrt(normDist[atom*3] ** 2 + normDist[atom*3+1] ** 2 + normDist[atom*3+2] ** 2))
        
        bin_wid = 20.0
        bins = np.arange(min(abs_vel), max(abs_vel) + bin_wid, bin_wid)
        _, bins, _ = plt.hist(abs_vel, bins=bins,  facecolor='r', alpha=0.2)    
        plt.xlabel('Начальная скорость (м/с)')
        plt.ylabel('Вероятность')
        plt.title('Распределение по скоростям')
        plt.grid()
    #    plt.show()
        plt.savefig('is_distr.png')  
        plt.close()

    def correct_mom(self):
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


    def run_simulation(self, step, numSteps):
        self.update_forces()
        self.verlet_integration()
        self.update_temperature()
        self.update_potentials()
        self.reset_forces()
        if (step+1) % 10 == 0:
            print("----------------- Прогресс: " + str(step+1) + "/" + str(numSteps) + " --------------------")
            print("Осталось (по моим расчетам, в минутах): " + str((1.7/60)*(numSteps-step)))

        if step > 20 and step < 120:
            self.scale_temperature()
        
    def update_forces(self):
        for atom1 in range(0, self.numAtoms-1):
            for atom2 in range(atom1+1, self.numAtoms):
                self.calculate_force(atom1, atom2)
        
        for atom in range(0, self.numAtoms):
            self.atoms[atom].fx *= 48*self.e
            self.atoms[atom].fy *= 48*self.e
            self.atoms[atom].fz *= 48*self.e
            self.atoms[atom].potential *= 4*self.e
            
    def calculate_force(self, atom1, atom2):
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
            
            # Обновляем силы
            self.atoms[atom1].fx += force*dx
            self.atoms[atom2].fx -= force*dx
            self.atoms[atom1].fy += force*dy
            self.atoms[atom2].fy -= force*dy
            self.atoms[atom1].fz += force*dz
            self.atoms[atom2].fz -= force*dz
            
            # Обновляем потенциалы
            self.atoms[atom1].potential += pot
            self.atoms[atom2].potential += pot

            self.p_int += -force * r2 / 2
            
    def verlet_integration(self):
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

    def reset_forces(self):
        for atom in range(0, self.numAtoms):
            self.atoms[atom].fx = 0
            self.atoms[atom].fy = 0
            self.atoms[atom].fz = 0
            self.atoms[atom].pot = 0
            
    def update_temperature(self):
        # Вычисление текущей температуры системы
        sumv2 = 0
        for atom in self.atoms:
            sumv2 += atom.vx**2 + atom.vy**2 + atom.vz**2
        self.currentTemp = (self.m/(3*self.numAtoms*self.kb))*sumv2
        self.temperatures.append(self.currentTemp)

    def update_potentials(self):
        # Вычисление текущей пот. энергии системы
        epot = 0
        for atom in self.atoms:
            epot += atom.potential
        self.potentials.append(epot)
    
    def get_atoms(self):
        return copy.deepcopy(self.atoms)
        
    def scale_temperature(self):
        if self.currentTemp > 100.0 or self.currentTemp < 80.0:
            print("Обновляем скорости...")
            for atom in range(0, self.numAtoms):
                self.atoms[atom].vx *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vy *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vz *= math.sqrt(self.temp/self.currentTemp)
    
    def get_pressure(self):
        # print(self.numAtoms / self.Volm * self.kb * self.currentTemp)
        # print(self.p_int / 6 / self.Volm)
        return (self.numAtoms / self.Volm * self.kb * self.currentTemp) - self.p_int / 6 / self.Volm * self.e * self.sigma


class FileWriter:
    
    def __init__(self):
        try:
            os.remove("temperatures.csv")
        except OSError:
            pass
        
        try:
            os.remove("rdf.csv")
        except OSError:
            pass

        try:
            os.remove("vac.csv")
        except OSError:
            pass
        
        try:
            os.remove("argon.xyz")
        except OSError:
            pass
        
        
        with open("argon.xyz", "a") as output:
            output.write("864\n") 
            output.write("Ar\n")

    def write_data(self, filename, data):
        with open(filename, "a") as output:
            for point in data:
                output.write("%s\n" % point)

    def write_xyz(self, atoms):
        with open("argon.xyz", "a") as output:
            for atom in atoms:
                output.write("Ar %s %s %s\n" % (atom.x, atom.y, atom.z))


class Analysis:
    
    consts = Constants()
    kb = consts.kb
    Na = consts.Nav
    sigma = consts.sigma
    dr = consts.dr
    eps = consts.e

    V = consts.V
    print('V =', V)
    numAtoms = consts.numAtoms
    dt = consts.dt

    lbox = consts.lbox
    
    originalAtoms = [] # Список для атомов
    currentAtoms = []
    nr = [] # число частиц на расстоянии r
    velacfinit = 0 # авто кор. скоростей в момент времени t=0
    velacf = 0 # в след.
    
    velaclist = [] # АКС для моделирования
    radiuslist = []
    timelist = []
    
    def __init__(self, atoms):
        self.originalAtoms = atoms
        
    def update_atoms(self, atoms):
        self.currentAtoms = atoms
    
    def pair_distribution_function(self):
        atom_counts = [0]*50
        cur_r = 0
        
        print("Создание радиального распределения..."),
        
        for atom1 in range(0, self.numAtoms - 1):
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


    def velocity_autocorrelation(self, step):
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
    
    def get_vac(self):
        return self.velaclist

    def plot_rdf(self):
        pass
        rdf = np.loadtxt("rdf.csv")
        for radius in range(0,50):
            self.radiuslist.append(radius * self.dr * (10 ** 10))
        plt.figure()
        plt.plot(self.radiuslist, rdf)
        plt.xlabel('r, А')
        plt.ylabel('Радиальное распределение')
        plt.show()
        

    def plot_vac(self, nSteps):
       vac = np.loadtxt("vac.csv")
       vac[0] = 1
       
       for time in range(0, nSteps):
            self.timelist.append(float(time) * self.dt * (10**12))
       
       plt.figure()
       plt.plot(self.timelist, vac)
       plt.xlabel('t, пс')
       plt.ylabel('Авто корреляция скоростей')
       plt.grid()
       plt.show()

    def plot_energy(self, temperatures, potentials, nSteps):

        KE = []
        for temp in temperatures:
            KE.append(3*self.numAtoms * self.kb * temp / 2)
        
        time_list = []
        for time in range(0, nSteps):
            time_list.append(float(time) * self.dt * (10**12))
        
        etot = []
        for energy in range(0, nSteps):
            etot.append(KE[energy] + potentials[energy])
            
        plt.figure()
        plt.plot(time_list, [pot / max(np.abs(etot)) for pot in KE], label='Екин')
        # potentials_inkj_per_mol = [pot * self.eps * self.Na / self.numAtoms * 1.0e-3 for pot in potentials]
        potentials_au = [pot / max(np.abs(etot)) for pot in potentials]
        plt.plot(time_list, potentials_au, label='Епот')
        plt.plot(time_list, [pot / max(np.abs(etot)) for pot in etot], label='Еполн')
        plt.xlabel('Время, пс')
        plt.ylabel('Энергия, au')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_temperature(self, temperatures, nSteps):
    
        time_list = []
        for time in range(nSteps):
            time_list.append(float(time) * self.dt * (10**12))
            
        plt.figure()
        plt.plot(time_list, temperatures)
        plt.xlabel('Время, пс')
        plt.ylabel('Температура, K')
        plt.grid()
        plt.show()