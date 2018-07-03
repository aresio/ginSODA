from differentiator import SystemsDifferentiator
from numpy import linspace, arange, loadtxt, savetxt, genfromtxt, array, concatenate, zeros
import platform
from subprocess import check_output
import os
#from pylab import *
import time

class ginSetup(object):

	def __init__(self, architecture=None, GPUs=[0]):
		self.SD = SystemsDifferentiator()
		self._GPUs = GPUs
		self.output = []
		self.cuda_compiler = "nvcc"
		self.temp_dir_objects = "TEMP"
		self.model_dir = "."
		self.output_dir = "."
		self.output_prefix = "sim"
		self.binary = "./a.out"
		self.machine_BITS = 64
		self._use_shared_memory = True
		if architecture!=None:
			self._architecture = architecture
		else:
			self._architecture = self.query_compute_capability()
		self.force_rebuild = False
		self.default_tpb = 32

		try:
			self.metadata = loadtxt("metadata", dtype="int")
			self.metadata[1]			
		except:
			self.metadata = [0,0]
			print " * ginSODA's meta-data not found."

		banner()

		print " * Assuming architecture SM_%s" % (self._architecture)
		print " * Detected available shared memory per SM: %d bytes" % (self.query_available_shared_memory())

	def query_available_shared_memory(self):
		import pycuda.autoinit
		import pycuda.driver as cuda
		for devicenum in range(cuda.Device.count()):
			device=cuda.Device(devicenum)
			attrs=device.get_attributes()
			return attrs[pycuda._driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
		

	def query_compute_capability(self):
		import pycuda.autoinit
		import pycuda.driver as cuda
		for devicenum in range(cuda.Device.count()):
			device=cuda.Device(devicenum)
			attrs=device.get_attributes()
			return "%d%d" % (attrs[pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MAJOR], attrs[pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MINOR])

	def check_reactions(self):
		print " * System of coupled ODEs:"
		for e,v in zip(self.SD.equations_rhs, self.SD.variables):
			print "\td("+v+")/dt\t = "+e

	def check_last_model(self, saved_lasthash):
		""" This method checks whether the model changed or not since the last simulation.
			If so, it returns True. Else, it returns False. It also returns the has of the model
		"""
		modelhash=self.SD.get_hash()
		if modelhash!=saved_lasthash:
			#print " * New model detected, ginSODA will recompile the simulator object (%d != %d)." % (modelhash, saved_lasthash)
			return True, modelhash
		return False, modelhash

	def check_last_access(self, saved_lastaccess):
		try:
			read_lastaccess = os.stat("__jitkernel.cu")[-2]
		except:
			print "ERROR: __jitkernel.cu not found, aborting"
			exit(1)		
		if saved_lastaccess<read_lastaccess:
			print " * The file '__jitkernel.cu' changed since last access, ginSODA must re-build the whole binary executable file."
			return True, read_lastaccess
		return False, read_lastaccess


	def set_model_dir(self, path):
		try:
			os.mkdir(path)
		except:
			pass
		finally:
			self.model_dir = path

	def set_output_dir(self, path):
		try:
			os.mkdir(path)
		except:
			pass
		finally:
			self.output_dir = path

	def set_output_prefix(self, pref):
		self.output_prefix = pref

	def create_model_files(self, parameters=[], initial_values=[], time_instants=[], observed=[]):

		savetxt(self.model_dir+"/c_matrix", parameters, delimiter="\t")
		savetxt(self.model_dir+"/MX_0", initial_values, delimiter="\t")
		savetxt(self.model_dir+"/t_vector", time_instants)
		savetxt(self.model_dir+"/cs_vector", array(observed), fmt="%d")
		print " * Model files created in", self.model_dir

	def create_kernel(self, path):
		with open(path, "w") as fo:
			fo.write(self.SD.create_LSODA_headers())
			print " * Kernel", path, "created"

	def compile(self, path_main, path_ginsoda, JUST_LINK=False, COMPILE_2B=False, verbose=True):

		print " * Checking out ginSODA's integration files, please wait..."

		FIRST_RUN, read_lastaccess = self.check_last_access(self.metadata[0])
		MODEL_CHANGED, modelhash   = self.check_last_model(self.metadata[1])	

		if FIRST_RUN:
			print " * First run ever"
			SKIP_COMPILE_1A = False
			COMPILE_2B = True
		elif MODEL_CHANGED:
			print " * Model changed since last simulation"
			SKIP_COMPILE_1A = True
			COMPILE_2B = True
			#print not JUST_LINK and not SKIP_COMPILE_1A
		else:
			#print " * Model good"
			SKIP_COMPILE_1A = True
			COMPILE_2B = False

		"""
		COMPILE_2B = False
		JUST_LINK = False
		SKIP_COMPILE_1A, read_lastaccess = self.check_last_access(self.metadata[0])

		if SKIP_COMPILE_1A: print " * ginSODA will not recompile the main simulator object."

		rec, modelhash = self.check_last_model(self.metadata[1])	
		if SKIP_COMPILE_1A: 
			COMPILE_2B = rec
			if COMPILE_2B:
				SKIP_COMPILE_1A = False

		print SKIP_COMPILE_1A, COMPILE_2B, JUST_LINK

		if self.force_rebuild:
			SKIP_COMPILE_1A = False
			JUST_LINK = False
			COMPILE_2B = False
		"""
		
		#exit()
		
		OPTIMIZATION_FLAGS = "-Ox"
		pltf=platform.system()

		if pltf=="Windows":
			OPTIMIZATION_FLAGS = "-Ox"
		elif pltf=="Linux":
			OPTIMIZATION_FLAGS = "-O3"

		try:
			os.mkdir(self.temp_dir_objects)
		except:
			#print "WARNING: directory", self.temp_dir_objects, "already exists"
			pass

		if COMPILE_2B:

			print " * Recompiling the simulation module"

			# CPU-side compilation of the simulator 
			command1 = [self.cuda_compiler, "--keep-dir", self.temp_dir_objects, "-maxrregcount=0", "--machine" , str(self.machine_BITS), "--compile",  "-Wno-deprecated-gpu-targets", 
			"--compiler-options", OPTIMIZATION_FLAGS, "-o", self.temp_dir_objects+"/"+path_ginsoda+".obj", path_ginsoda, "-clean", "-w"]
			if verbose: print "[Step #1]", " ".join(command1)
			ret = check_output(command1)
			#print ret
			

			# GPU-side compilation of the simulator
			# command3 = [self.cuda_compiler, "-gencode=arch=compute_"+self._architecture+",code=\"sm_"+self._architecture+",compute_"+self._architecture+"\"", 
			command3 = [self.cuda_compiler, "-arch=sm_"+self._architecture, 
		    "--keep-dir", self.temp_dir_objects, "-maxrregcount=0", "--machine", str(self.machine_BITS), "--compile", "-cudart", "static", "-use_fast_math", "-Wno-deprecated-gpu-targets",   
	    	"--compiler-options", OPTIMIZATION_FLAGS, "-o", self.temp_dir_objects+"/"+path_ginsoda+".obj", path_ginsoda, "-w"]
			if verbose: print "[Step #2]", " ".join(command3)
			ret = check_output(command3)

			#print ret		

			if not SKIP_COMPILE_1A:

				print " * Recompiling the whole ginSODA"

				# CPU-side compilation of the main
				command2 = [self.cuda_compiler, "--keep-dir", self.temp_dir_objects, "-maxrregcount=0", 
				"--machine",  str(self.machine_BITS), "--compile",  "-Wno-deprecated-gpu-targets", 
				"--compiler-options", OPTIMIZATION_FLAGS,
				# -D_MBCS -Xcompiler \"/EHsc /W0 /nologo /O2 /Zi  /MD \"
				"-o", self.temp_dir_objects+"/"+path_main+".obj", path_main, "-clean", "-w"]
				if verbose: print "[Step #3]", command2
				ret = check_output(command2)
				print ret
				
				# GPU-side compilation of the main
				# command4 = [self.cuda_compiler, "-gencode=arch=compute_"+self._architecture+",code=\"sm_"+self._architecture+",compute_"+self._architecture+"\"",
				command4 = [self.cuda_compiler, "-arch=sm_"+self._architecture,
				 #"-I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5\\include\" 
				"--keep-dir", self.temp_dir_objects, "-maxrregcount=0",			
				"--machine", str(self.machine_BITS), "--compile", "-cudart", "static", "-use_fast_math", "-Wno-deprecated-gpu-targets", 
				"--compiler-options", OPTIMIZATION_FLAGS,
				#  -D_MBCS -Xcompiler \"/EHsc /W0 /nologo /O2 /Zi /MD \"
				"-o", self.temp_dir_objects+"/"+path_main+".obj", path_main, "-w"]
				if verbose: print "[Step #4]",  command4
				ret = check_output(command4)
				#print ret	
			

		if COMPILE_2B: # that is, just link
			# link process
			commands5 = [self.cuda_compiler, "-arch=sm_"+self._architecture, self.temp_dir_objects+"/"+path_main+".obj", 
			self.temp_dir_objects+"/"+path_ginsoda+".obj", 
			"--machine",  str(self.machine_BITS),
			# "-Xcompiler", "\"/MD /Ox\"", 
			 "-o", self.binary, "-w"]
			if verbose: print " ".join(commands5)
			ret = check_output(commands5)
			# print ret

		savetxt("metadata", [read_lastaccess, modelhash], fmt="%d")
		print " * ginSODA simulator successfully built."	


	def create_ginsoda_files(self, lsoda_settings):

		try:
			savetxt(self.model_dir+"/rtol", [lsoda_settings["rtol"]], delimiter="\n")
		except:
			print "WARNING: no relative tolerance specified, using default 1e-6"
		finally:
			print " * Relative tolerances file created"

		try:
			savetxt(self.model_dir+"/atol_vector", lsoda_settings["atol_vector"], delimiter="\n")
		except:
			print "WARNING: no absolute tolerance vector specified, using default 1e-12"
		finally:
			print " * Absolute tolerances file created"


		try:
			savetxt(self.model_dir+"/max_steps", [lsoda_settings["max_steps"]], delimiter="\n", fmt="%d")
		except:
			print "WARNING: no max steps specified, using default 10000"
		finally:
			print " * Max internal steps file created"



	def run(
			self, parameters=[], initial_values=[], time_instants=[], GPU=0,  
			observed=[], lsoda_settings={}, no_simulation=False, dump_cuda_output=False):		

		start_total_time = time.time()

		# check simulation configuration
		if len(parameters)!=len(initial_values):			
			raise Exception("parameterizations and initial states have different lengths ("+str(len(parameters))+" and "+str(len(initial_values))+ ", respectively)")

		if len(parameters[0])!=len(self.SD.parameters):
			raise Exception("number of parameters per thread is different with respect to the number of parameters")

		if len(initial_values[0])!=len(self.SD.variables):
			raise Exception("number of initial values per thread is different with respect to the number of parameters")
		
		if len(observed)==0: observed=arange(len(initial_values[0])) # sample all species if observed are not specified 
		print " * Observed variables indices:", observed

		self.differentiate_everything(verbose=False)

		rec, modelhash = self.check_last_model(self.metadata[1])	

		start_files_creation = time.time()

		if rec: self.create_kernel("__jitkernel.cu")
		self.create_model_files(parameters=parameters, initial_values=initial_values, time_instants=time_instants, observed=observed)
		self.create_ginsoda_files(lsoda_settings)
		if no_simulation: return

		end_files_creation = time.time()

		print "[t] Files creation required %f seconds" % (end_files_creation - start_files_creation)

		start_compile_time = time.time()

		self.compile("kernel.cu", "lsoda.cu")

		end_compile_time = time.time()

		print "[t] Compilation completed in %f seconds" % (end_compile_time - start_compile_time)

		threads, blocks = self.organize_work(len(initial_values))		

		start_launch = time.time()
		
		ret = self.launch(threads, blocks, GPU)
		
		end_launch = time.time()

		print "[t] Simulation completed in %f seconds" % (end_launch - start_launch)

		end_total_time = time.time()

		print "[t] Overall execution completed in %f seconds" % (end_total_time - start_total_time)

		if dump_cuda_output:
			print " * Output returned by the CUDA binary:"
			print "*"*80
			print ret
			print "*"*80

		return self.organize_results(ret, time_instants)


	def organize_results(self, result, time_instants):
		all_dynamics = []
		full_riga = []
		for riga in result.split("\n"):

			if riga.strip()=="": 
				full_riga = array(full_riga)
				species = len(full_riga)/len(time_instants)
				all_dynamics.append(full_riga.reshape((len(time_instants),species)))
				full_riga = []			
			else:
				pr = map(float, riga.split())
				full_riga.extend(pr)
		return all_dynamics

	def organize_work(self, par):
		""" This method calculates the heuristic for the 
			threads / blocks subdivision.
		"""
		N = par
		T = 0
		B = T = 0
		if N<self.default_tpb:
			B=1
			T=N
		elif N%self.default_tpb==0:
			B=N/self.default_tpb
			T=self.default_tpb
		else:
			B=N/self.default_tpb+1
			T=self.default_tpb
		return T,B

	def launch(self, threads, blocks, GPU=0):
		# launch simulation		

		if self._use_shared_memory:
			print " * Using shared memory"
			sh = "1" 
		else: 
			print " * Not using shared memory"
			sh = "0"

		print " * Launching job: %d threads subdivided into %d blocks" % (threads*blocks, blocks)
		print " * Launching model in", self.model_dir
		command = [self.binary, "-i", self.model_dir,  "-o", self.output_dir, "-p", self.output_prefix,
		"-b", str(blocks), "-g", str(GPU), "-d", "0", "-h", sh, "-t", "1"]
		ret = check_output(command)
		return ret

	def specify_model(self, variables=None, parameters=None, equations=None):
		if variables==None:	raise Exception("Variables not specified, aborting.")
		if parameters==None: raise Exception("Parameters not specified, aborting.")
		if equations==None:	raise Exception("Equations not specified, aborting.")
		self.add_variables(variables)
		self.add_parameters(parameters)
		self.add_equations(equations)
		print " * Model loaded"

	def add_variables(self, variables):
		for v in variables:
			if len(v)<2: raise Exception("Please use names longer than a single character ("+v+" is not valid)")				
			self.SD.variables.append(v)

	def add_parameters(self, parameters):
		for c in parameters:
			self.SD.parameters.append(c)

	def add_equations(self, reactions):
		for r in reactions:
			self.SD.equations_rhs.append(r)

	def clear(self):
		self.variables.clear()
		self.parameters.clear()
		self.output.clear()

	def differentiate_everything(self, verbose=False):
		self.SD.differentiate_everything(verbose=verbose)

	def estimate_memory_requirements(self, THREADS, time_instants):
		"""
			Calculate the exact amount of memory used for the simulations.
		"""
		var_bytes =  len(self.SD.variables)*8*THREADS*2
		params_bytes = len(self.SD.parameters)*8*THREADS
		jacob_bytes = len(self.SD.variables)**2*8*THREADS
		lsoda_bytes = THREADS*(4*8 + 3*8 + 8*len(self.SD.variables))
		lrw = 22+len(self.SD.variables)*max(16, len(self.SD.variables)+9); rwork_bytes = THREADS*8*lrw
		liw = 20+len(self.SD.variables); iwork_bytes = THREADS*liw*4
		output_bytes = len(time_instants)*THREADS*8*len(self.SD.variables)
		commonblock = 4304*THREADS*8
		total = var_bytes + params_bytes + jacob_bytes + rwork_bytes + lsoda_bytes + iwork_bytes + output_bytes + commonblock


		print " Dimension of the system:"
		print " - variables:", len(self.SD.variables), "\tmemory used:", var_bytes, "bytes"
		print " - parameters:", len(self.SD.parameters), "\tmemory used:", params_bytes, "bytes"
		print " - jacobian:", len(self.SD.variables)**2, "elements\tmemory used:", jacob_bytes, "bytes"
		print " - service LSODA structures:", lsoda_bytes, "bytes"
		print " - rwork: ", rwork_bytes, "bytes"
		print " - iwork: ", iwork_bytes, "bytes"
		print " - output dynamics", output_bytes, "bytes"
		print " - common block:", commonblock, "bytes"
		print " - TOTAL estimated global memory requirements:", total/1e6, "MB for %d threads" % (THREADS)
		estimated_sm = self.estimate_shared_memory_requirements(THREADS, len(self.SD.variables))
		print " - Shared memory requirements: %d bytes per block" % (estimated_sm)
		if estimated_sm<=self.query_available_shared_memory():
			print "   ginSODA automatically ENABLED shared memory support"
			self._use_shared_memory = True
		else:
			print "   ginSODA automatically DISABLED shared memory support"
			self._use_shared_memory = False
		return total


	def estimate_shared_memory_requirements(self, THREADS, n_variables):
		t, b = self.organize_work(THREADS)
		return 8*t*(n_variables+1)  # per block



def banner():
	print "        _      ________  ___  ___ "
	print "  ___ _(_)__  / __/ __ \/ _ \/ _ |"
	print " / _ `/ / _ \_\ \/ /_/ / // / __ |"
	print " \_, /_/_//_/___/\____/____/_/ |_|"
	print "/___/                version 0.9.0"
	print 

if __name__ == '__main__':

	pass