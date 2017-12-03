##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################


import numpy as np 

params = np.load('BiLSTM_parameters.npy').item()


"""_-_-_-_-_-_-_-_-_-_-_-_LAYER 1_-_-_-_-_-_-_-_-_-_-_-_-_"""


class layer1(object):

	#### FORWARD ######
	class forward(object):
		def __init__(self):
			self.Wix = np.array(params['W_i_x_fw_layer1']) 
			self.Wih = np.array(params['W_i_h_fw_layer1']) 
			self.Wic = np.array(params['peephole_i_c_bw_layer1']) 
			self.bi = np.array(params['b_i_fw_layer1']) 

			self.Wfx = np.array(params['W_f_x_fw_layer1']) 
			self.Wfh = np.array(params['W_f_h_fw_layer1']) 
			self.Wfc = np.array(params['peephole_f_c_bw_layer1'])
			self.bf = np.array(params['b_f_fw_layer1'])

			self.Wcx = np.array(params['W_c_x_fw_layer1']) 
			self.Wch = np.array(params['W_c_h_fw_layer1']) 
			self.bc = np.array(params['b_c_fw_layer1']) 

			self.Wox = np.array(params['W_o_x_fw_layer1']) 
			self.Woh = np.array(params['W_o_h_fw_layer1']) 
			self.Woc = np.array(params['peephole_o_c_bw_layer1']) 
			self.bo = np.array(params['b_o_fw_layer1'])
			##seeding the states
			
			self.i = np.random.uniform(-1, 1, size=(207,320)).astype(np.float32)
			# self.hidden_state = np.random.uniform(-1, 1, size=(207,320))
			# self.cell_state = np.random.uniform(-1, 1, size=(207,320))
			self.hidden_state = np.zeros((207,320)).astype(np.float32)
			self.cell_state = np.zeros((207,320)).astype(np.float32)
			self.output = np.ndarray(shape=(207, 320)).astype(np.float32)
			self.f = np.ndarray(shape=(207, 320)).astype(np.float32)




	#### BACKWARD #####
	class backward(object):
		def __init__(self):
			self.Wix = np.array(params['W_i_x_bw_layer1']) 
			self.Wih = np.array(params['W_i_h_bw_layer1']) 
			self.Wic = np.array(params['peephole_i_c_bw_layer1']) 
			self.bi = np.array(params['b_i_bw_layer1']) 

			self.Wfx = np.array(params['W_f_x_bw_layer1']) 
			self.Wfh = np.array(params['W_f_h_bw_layer1']) 
			self.Wfc = np.array(params['peephole_f_c_bw_layer1'])
			self.bf = np.array(params['b_f_bw_layer1'])

			self.Wcx = np.array(params['W_c_x_bw_layer1']) 
			self.Wch = np.array(params['W_c_h_bw_layer1']) 
			self.bc = np.array(params['b_c_bw_layer1']) 

			self.Wox = np.array(params['W_o_x_bw_layer1']) 
			self.Woh = np.array(params['W_o_h_bw_layer1']) 
			self.Woc = np.array(params['peephole_o_c_bw_layer1']) 
			self.bo = np.array(params['b_o_bw_layer1'])

			self.i = np.random.uniform(-1, 1, size=(207,320)).astype(np.float32)
			# self.hidden_state = np.random.uniform(-1, 1, size=(207,320))
			# self.cell_state = np.random.uniform(-1, 1, size=(207,320))
			self.hidden_state = np.zeros((207,320)).astype(np.float32)
			self.cell_state = np.zeros((207,320)).astype(np.float32)
			self.output = np.ndarray(shape=(207, 320)).astype(np.float32)
			self.f = np.ndarray(shape=(207, 320)).astype(np.float32)


"""_-_-_-_-_-_-_-_-_-_-_-_LAYER 2_-_-_-_-_-_-_-_-_-_-_-_-_"""


class layer2(object):
	
	#### FORWARD ######
	class forward(object):
		def __init__(self):
			self.Wix = np.array(params['W_i_x_fw_layer2']) 
			self.Wih = np.array(params['W_i_h_fw_layer2']) 
			self.Wic = np.array(params['peephole_i_c_bw_layer2']) 
			self.bi = np.array(params['b_i_fw_layer2']) 

			self.Wfx = np.array(params['W_f_x_fw_layer2']) 
			self.Wfh = np.array(params['W_f_h_fw_layer2']) 
			self.Wfc = np.array(params['peephole_f_c_bw_layer2'])
			self.bf = np.array(params['b_f_fw_layer2'])

			self.Wcx = np.array(params['W_c_x_fw_layer2']) 
			self.Wch = np.array(params['W_c_h_fw_layer2']) 
			self.bc = np.array(params['b_c_fw_layer2']) 

			self.Wox = np.array(params['W_o_x_fw_layer2']) 
			self.Woh = np.array(params['W_o_h_fw_layer2']) 
			self.Woc = np.array(params['peephole_o_c_bw_layer2']) 
			self.bo = np.array(params['b_o_fw_layer2'])

			self.i = np.random.uniform(-1, 1, size=(207,320)).astype(np.float32)
			# self.hidden_state = np.random.uniform(-1, 1, size=(207,320))
			# self.cell_state = np.random.uniform(-1, 1, size=(207,320))
			self.hidden_state = np.zeros((207,320)).astype(np.float32)
			self.cell_state = np.zeros((207,320)).astype(np.float32)
			self.output = np.ndarray(shape=(207, 320)).astype(np.float32)
			self.f = np.ndarray(shape=(207, 320)).astype(np.float32)

	#### BACKWARD #####
	class backward(object):
		def __init__(self):
			self.Wix = np.array(params['W_i_x_bw_layer2']) 
			self.Wih = np.array(params['W_i_h_bw_layer2']) 
			self.Wic = np.array(params['peephole_i_c_bw_layer2']) 
			self.bi = np.array(params['b_i_bw_layer2']) 

			self.Wfx = np.array(params['W_f_x_bw_layer2']) 
			self.Wfh = np.array(params['W_f_h_bw_layer2']) 
			self.Wfc = np.array(params['peephole_f_c_bw_layer2'])
			self.bf = np.array(params['b_f_bw_layer2'])

			self.Wcx = np.array(params['W_c_x_bw_layer2']) 
			self.Wch = np.array(params['W_c_h_bw_layer2']) 
			self.bc = np.array(params['b_c_bw_layer2']) 

			self.Wox = np.array(params['W_o_x_bw_layer2']) 
			self.Woh = np.array(params['W_o_h_bw_layer2']) 
			self.Woc = np.array(params['peephole_o_c_bw_layer2']) 
			self.bo = np.array(params['b_o_bw_layer2'])

			self.i = np.random.uniform(-1, 1, size=(207,320))
			# self.hidden_state = np.random.uniform(-1, 1, size=(207,320))
			# self.cell_state = np.random.uniform(-1, 1, size=(207,320))
			self.hidden_state = np.zeros((207,320)).astype(np.float32)
			self.cell_state = np.zeros((207,320)).astype(np.float32)
			self.output = np.ndarray(shape=(207, 320)).astype(np.float32)
			self.f = np.ndarray(shape=(207, 320)).astype(np.float32)


# for k in params:
# 	if (k == '__version__')| (k== '__header__')|(k=='__globals__'):
# 		print k
# 	else :
# 		print k,':', np.array(np.array(params[k])).shape
#              
#          
## Reference shapes
# W_o_x_fw_layer1 : (320, 360)
# W_o_x_fw_layer2 : (320, 640)
# W_i_h_bw_layer2 : (320, 320)
# W_i_h_bw_layer1 : (320, 320)
# b_c_fw_layer1 : (1, 320)
# b_c_fw_layer2 : (1, 320)
# b_o_fw_layer1 : (1, 320)
# W_c_x_fw_layer2 : (320, 640)
# b_o_fw_layer2 : (1, 320)
# W_f_h_fw_layer2 : (320, 320)
# W_i_h_fw_layer2 : (320, 320)
# W_i_h_fw_layer1 : (320, 320)
# W_f_h_fw_layer1 : (320, 320)
# W_i_x_bw_layer2 : (320, 640)
# W_i_x_bw_layer1 : (320, 360)
# W_o_h_fw_layer1 : (320, 320)
# W_f_x_fw_layer2 : (320, 640)
# W_i_x_fw_layer2 : (320, 640)
# W_i_x_fw_layer1 : (320, 360)
# W_f_x_fw_layer1 : (320, 360)
# W_c_x_bw_layer1 : (320, 360)
# W_c_x_bw_layer2 : (320, 640)
# Fully_connected_layerweights : (44, 640)
# W_f_x_bw_layer2 : (320, 640)
# W_f_x_bw_layer1 : (320, 360)
# W_c_x_fw_layer1 : (320, 360)
# peephole_o_c_fw_layer1 : (1, 320)
# W_c_h_bw_layer1 : (320, 320)
# peephole_o_c_fw_layer2 : (1, 320)
# W_o_h_fw_layer2 : (320, 320)
# Fully_connected_layerbiases : (1, 44)
# W_c_h_fw_layer1 : (320, 320)
# W_c_h_fw_layer2 : (320, 320)
# b_c_bw_layer1 : (1, 320)
# peephole_f_c_bw_layer2 : (1, 320)
# peephole_f_c_bw_layer1 : (1, 320)
# b_c_bw_layer2 : (1, 320)
# W_o_x_bw_layer1 : (320, 360)
# peephole_f_c_fw_layer2 : (1, 320)
# peephole_f_c_fw_layer1 : (1, 320)
# W_o_x_bw_layer2 : (320, 640)
# b_o_bw_layer1 : (1, 320)
# b_o_bw_layer2 : (1, 320)
# b_i_fw_layer1 : (1, 320)
# W_f_h_bw_layer2 : (320, 320)
# b_f_fw_layer1 : (1, 320)
# b_f_fw_layer2 : (1, 320)
# W_f_h_bw_layer1 : (320, 320)
# b_f_bw_layer1 : (1, 320)
# b_f_bw_layer2 : (1, 320)
# peephole_i_c_fw_layer2 : (1, 320)
# W_c_h_bw_layer2 : (320, 320)
# b_i_fw_layer2 : (1, 320)
# peephole_i_c_bw_layer2 : (1, 320)
# peephole_o_c_bw_layer1 : (1, 320)
# b_i_bw_layer2 : (1, 320)
# b_i_bw_layer1 : (1, 320)
# peephole_o_c_bw_layer2 : (1, 320)
# peephole_i_c_fw_layer1 : (1, 320)
# W_o_h_bw_layer1 : (320, 320)
# peephole_i_c_bw_layer1 : (1, 320)
# W_o_h_bw_layer2 : (320, 320)
