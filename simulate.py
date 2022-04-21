from classes.simulation import Simulation
from classes.analysis import Analysis
from classes.filewriter import fileWriter
import sys


nSteps = 100

# Симулятор
sim = Simulation()
# Анализатор
analysis = Analysis(sim.getAtoms())
# Писатель
fw = fileWriter()


for step in range(0, nSteps):
    sim.runSimulation(step, nSteps)
    analysis.updateAtoms(sim.getAtoms())
    analysis.velocityAutocorrelation(step)
    fw.writeXYZ(sim.getAtoms())
    

fw.writeData("temp.csv", sim.temperatures)
fw.writeData("rdf.csv", analysis.pairDistributionFunction())
fw.writeData("vac.csv", analysis.getVAC())

analysis.plotRDF()
analysis.plotVAC(nSteps)
analysis.plotEnergy(sim.temperatures, sim.potentials, nSteps)
analysis.plotTemperature(sim.temperatures, nSteps)
