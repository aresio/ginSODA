from sympy import *
import hashlib

class SystemsDifferentiator(object):

	def __init__(self):
		self.equations_rhs = []
		self.variables     = []
		self.parameters    = []
		self.jacobian      = []

	def get_hash(self, type="normal"):
		if 		type=="normal": return hash(self.create_LSODA_headers()) 
		elif 	type=="sha512": return hashlib.sha512(self.create_LSODA_headers()).hexdigest() # TODO

	def replace_constants(self, d):
		while(1):
			found = False
			for c in self.parameters:
				if c in d:
					replace_with = "k["+str(self.parameters.index(c))+"]"					
					d = d.replace(c, replace_with)
					found = True
			if not found: return d
	
	def replace_variables(self, d):
		while(1):
			found = False
			for c in self.variables:
				if c in d:
					replace_with = "y["+str(self.variables.index(c))+"]"					
					d = d.replace(c, replace_with)
					found = True
			if not found: return d

	def replace_powers(self, d):
		#d = print_ccode(d)
		return d

	def differentiate_everything(self, verbose=True):
		self.jacobian = []
		
		if verbose:
			print "System of ODEs:"
			for m, equation in enumerate(self.equations_rhs):
				print "f_"+str(m)+" = "+ equation
			print
		
		if verbose:
			print "Jacobian matrix:"
		
		for m, equation in enumerate(self.equations_rhs):			
			for n, v in enumerate(self.variables):
				#print diff(equation, v), equation, v
				derivative = ccode(str(diff(equation,v)))
				derivative = self.replace_constants(derivative)
				derivative = self.replace_variables(derivative)
				#derivative = self.replace_powers(derivative)
				# res = "pd["+str(m)+"]["+str(n)+"] = " + derivative
				res = "pd["+str(m*len(self.variables)+n)+"] = " + derivative
				self.jacobian.append(res)
				if verbose:
					ver = "df_"+str(m)+"/d"+v+" = "+ derivative
					print ver,"\t",
			if verbose: print


	def create_LSODA_headers(self):
		# derivatives
		ret  = "\nextern \"C\" __device__ void  myFex(int *neq, double *t, double *y, double *ydot, double* k) {\n"
		ret += "\t	\n"
		for e,v in zip(self.equations_rhs, self.variables):
			e = ccode(e)
			e = self.replace_constants(e)
			e = self.replace_variables(e)
			ret += "\t\tydot["+ str(self.variables.index(v)) + "] = "+e+";\n"
		ret += "\t}\n"
		#ret += "};\n\n"

		# jacobian
		ret += "extern \"C\" __device__ void  myJex(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd, double* k) {\n"
		ret += "\t \n"
		for e in self.jacobian:
			ret += "\t\t" + e + ";\n"
		ret += "\t}\n"
		#ret += "};\n\n"
		return ret




if __name__ == '__main__':
	pass
