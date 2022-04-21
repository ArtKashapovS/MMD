from package import Simulation, Analysis, FileWriter


n_steps = 100

# Симулятор
sim = Simulation()
# Анализатор
analysis = Analysis(sim.get_atoms())
# Писатель
fw = FileWriter()


for step in range(0, n_steps):
    sim.run_simulation(step, n_steps)
    analysis.update_atoms(sim.get_atoms())
    analysis.velocity_autocorrelation(step)
    fw.write_xyz(sim.get_atoms())
    

fw.write_data("temp.csv", sim.temperatures)
fw.write_data("rdf.csv", analysis.pair_distribution_function())
fw.write_data("vac.csv", analysis.get_vac())

analysis.plot_rdf()
analysis.plot_vac(n_steps)
analysis.plot_energy(sim.temperatures, sim.potentials, n_steps)
analysis.plot_temperature(sim.temperatures, n_steps)
