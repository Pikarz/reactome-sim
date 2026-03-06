import roadrunner

# SBML model
rr = roadrunner.RoadRunner("~/Study/technopole/homo_sapiens.3.1.sbml/R-HSA-1855192.sbml")
working_rr = roadrunner.RoadRunner("~/Study/technopole/working_examples/00001-sbml-l3v1.xml")

# simulate from 0 to 10 time units with 100 output rows
result = rr.simulate(0,10,100)
result2 = working_rr.simulate(0, 10, 100)

rr.plot()

working_rr.plot()


