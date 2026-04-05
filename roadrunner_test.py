import roadrunner
import os
import pylab

# Build absolute paths relative to this script's location
_here = os.path.dirname(os.path.abspath(__file__))

# SBML model
rr = roadrunner.RoadRunner(os.path.join(_here, "working_homo-sapiens", "R-HSA-1855192.sbml"))
#working_rr = roadrunner.RoadRunner(os.path.join(_here, "working_examples", "00001-sbml-l3v1.xml"))

# simulate from 0 to 10 time units with 100 output rows
result = rr.simulate(0,10000,100)
#result2 = working_rr.simulate(0, 10, 100)

# plot primo modello
for i in range(1, result.shape[1]):
    pylab.plot(result[:,0], result[:,i], label=result.colnames[i])

pylab.xlabel("Time")
pylab.ylabel("Concentration")
pylab.title("Simulation")
pylab.legend()

pylab.show()


