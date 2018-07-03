from ginsoda import *
from pylab import *

if __name__ == '__main__':
	

	# Example of Enzyme Kinetics 
	GS = ginSetup(architecture="21")
	GS.add_variables(["Substrate", "Enzyme", "Complex", "Product"])
	GS.add_parameters(["k0", "k1", "z2"])
	GS.add_equations([
		"-k0*Substrate*Enzyme+k1*Complex",        # S:  S+E->ES, ES->S+E
		"-k0*Substrate*Enzyme+k1*Complex+z2*Complex",  # E:  S+E->ES, ES->S+E, ES -> P+E
		"+k0*Substrate*Enzyme-k1*Complex-z2*Complex",  # ES: S+E->ES, ES -> S+E, ES -> P+E
		"+z2*Complex" ])         	  # P:  ES -> P+E
	GS.set_model_dir("MODELTEST")
	GS.set_output_dir("OUTPUTDIR")
	GS.set_output_prefix("PREFIX")
	GS.check_reactions()	
	GS.force_rebuild=False

	THREADS = 4096


	#parameters     = [[0.0025,0.1,5.0]]*THREADS
	parameters      = []
	perturbation    = linspace(2.5e-3, 2.5e-2, THREADS)
	for x in xrange(THREADS):
		parameters.append( [ perturbation[x], 0.1, 5.0] )

	initial_values = [[1000,500,0,0]]*THREADS
	# initial_values = arange(2*THREADS).reshape(THREADS,2)
	time_instants  = linspace(0,5,50)
	atol_vector    = [[1e-6]]*THREADS
	rtol 		   = 1e-4
	max_steps      = 500

	# GS.estimate_memory_requirements(THREADS, time_instants)
	#exit()

	#GS._use_shared_memory = True

	all_dynamics = GS.run(
		lsoda_settings={'max_steps': max_steps, 'atol_vector':atol_vector, 'rtol': rtol},
		parameters=parameters, 
		initial_values=initial_values,
		time_instants=time_instants,		
		no_simulation=False
	)
	
	#exit()
	for s in xrange(4):
		plot(all_dynamics[0].T[0], all_dynamics[0].T[s+1], label=GS.SD.variables[s]+"_model1")
	for s in xrange(4):
		plot(all_dynamics[32].T[0], all_dynamics[32].T[s+1], "--", label=GS.SD.variables[s]+"_model2")
	legend(ncol=2)
	show()
