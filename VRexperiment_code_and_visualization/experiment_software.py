""" 
this script implements the logic of the OBE experiment
if run as main, it will enter simulation mode and plot 3D obe trajectories 

Created on Thu Feb 15 10:11:26 2018
@author: nesti
"""


from skopt import Optimizer
import pickle
import csv
import socket

try:
    from base_module import BaseModule
except:
    class BaseModule:
        pass
    
    
def getClassName():
    return "Experiment"

def getFuncNames():
    return ""
       
class Experiment(BaseModule):

    def __enter__(self):
        return self
            
    def reset(self, core):
        BaseModule.reset(self,core)
        self.exp_state = 0 

    def __exit__(self, exc_type, exc_value, traceback):
        print('Experiment over')
        
    def __init__(self,  simulation = 0):        
        self.is_simulation = simulation 
        self.exp_state = 0
        try:
            with open('exp_data_incomplete.pkl', 'rb') as f:
                self.opt = pickle.load(f)
        except:
            self.opt = Optimizer(base_estimator = 'GP', acq_func = 'LCB', acq_optimizer = 'auto',
                             dimensions        = [(0.0, 4.0),   # range for param 1 (eg trajectory final height?)
                                                  (1.0, 5.0),   # range for param 2 (eg trajectory final pitch?)
                                                  (1.0, 5.0),   # range for param 3 (eg audio gain?)
                                                  (-2.0,2.0)],  # range for param 4 (eg vibrational state duration?)
                            acq_func_kwargs    = {'kappa': 5},  # we should prefer explore. howver, with higher dim it will naturally tend to diversify, so kappa could be decreased
                            n_initial_points   = 10
                            )
            # NB pitching 90 is problematic for the quaternion, debug required. for now, range should be [0-89]
        
        if not self.is_simulation:
            BaseModule.__init__(self)

    def update(self, time):
        # behave according to the state of the experiment
        if self.exp_state == 0:                                                 # set new parameters
            self.new_stim = self.opt.ask()
            self.set_new_OBE(self.new_stim, time)                               
            print('new OBE parameters:', self.new_stim)
            self.exp_state = 1

        elif self.exp_state == 1:                                               # waiting for button press
            print('Waiting for the participant to press a button and start the trial')
            pressed = True # pressed = read_button_state() 
            if pressed == True:
                self.exp_state = 2

        elif self.exp_state == 2:                                               # playing trial
            stop = self.play_OBE(time)
            if stop == True:
                self.exp_state = 3

        elif self.exp_state == 3:                                               # waiting for answer
            print('Waiting for participant\'s answer')            
            answer_recorded = self.get_answer()
            if answer_recorded == True:
                self.exp_state = 0            
            
    def read_param(self):
        # this function is used for udp communicaiton with matlab, 
#        not necessary if we switch to skopt (python implementation of bayesian optimization)
        data, self.addr = self.sock.recvfrom(1024); 
        data = self.realism.get_str_parameter(self.core, "udp", "lastMessage")
        return data
        
    def set_new_OBE(self, new_param, time):
        if self.is_simulation: 
            # plot 
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import numpy as np
            
            global ax
            fig1 = plt.figure()
            ax = fig1.gca(projection='3d') 
            ax.set_xlabel('x'), ax.set_xlim([-2, 2]), ax.set_ylabel('z') 
            ax.set_ylim([-2, 2]),ax.set_zlabel('y'), ax.set_zlim([-2, 2])

            # inizialize camera position
            lastPos     =  [0., 0., 0.]
            lastDirect  =  [1., 0., 0.]
            lastUp      =  [0., 1., 0.]
            lastRight   =  np.cross(lastUp,lastDirect)
            # apparently, the rotation matrix can be simply constructed from the above vectors
            # check with FLORIAN, also when using realism compare this with the get_camera function
            lastT       =  np.vstack((np.vstack((lastRight, lastUp, lastDirect, lastPos)).T, [0,0,0,1]))

        else:
            lastPos     =  self.realism.get_camera_position(self.core)
            lastDirect  =  self.realism.get_camera_direction(self.core)
            lastUp      =  self.realism.get_camera_up(self.core)
            lastT       =  self.realism.get_camera_transformationMatrix(self.core) # check correct realism function
            
        extraction_duration, extraction_angle, final_pitch, vibState_duration = new_param # !! make sure to be consistent with opt inizialization !!
        extraction_duration, extraction_angle, final_pitch, vibState_duration =  [7.,-45.,-40.,0.]  # ONLY NOW FOR DEBUG, COMMENT OUT.
        self.new_OBE  = OBE(start_pos           = lastPos, 
                            start_direct        = lastDirect, 
                            start_Up            = lastUp,
                            start_T             = lastT, 
                            start_time          = time, 
                            extraction_duration = extraction_duration, 
                            height              = 1.5,
                            final_pitch         = final_pitch,
                            finalOffset         = 0,
                            extractionAngle     = extraction_angle,
                            vibState_duration   = vibState_duration, 
                            appmove_duration    = 1,
                            sound_type          = 0, 
                            sound_traj_gain     = 0
                            )

    def play_OBE(self, time):
        self.new_OBE.update(time)           # updating relevant OBE attributes. next we retireve them and pass them to realism
        
        T       = self.new_OBE.the_self.T
        
        # all we should need should be T, but if necessary we can retrieve everything else
        pos         = self.new_OBE.the_self.pos
        direct      = self.new_OBE.the_self.direct
        UP          = self.new_OBE.the_self.UP
        LAP         = self.new_OBE.the_self.LAP
        sound_pos   = self.new_OBE.sound_source.pos

        if self.is_simulation:
            self.new_OBE.animate(pos, LAP, UP, sound_pos)
            with open('view_matrices.csv', 'ab') as csvfile:   # export the view matrix to debug realism. NB: will append to existing file, so rename afterwards if necessary
                np.savetxt(csvfile,T , fmt = '%5.5f', newline = '\n', delimiter = " ")

        else:
            # using transformation matrix
            self.realism.setTransformationMatrix(self.core, T)      # check for the right instruction
            self.realism.set_Sound(self.core, self.new_OBE.sound_source.pos, self.new_OBE.sound_source.sound_type)      # check for the right instruction
            
            # or, we can use lookat function
#            pos = pos - self.new_OBE.the_body.pos                                   # probably only necessary because of a realism bug, we'll have to get rid of this
#            self.realism.set_camera_look_at(self.core, pos[0], pos[1], pos[2],      # camera position
#                                            LAP[0], LAP[1], LAP[2],                 # look-at point
#                                            UP[0], UP[1], UP[2])                    # up vector wrt camera 
        
        return True if self.new_OBE.obe_state == 2 else False    # returns true when OBE is over

    def get_answer(self):
        if self.is_simulation:
            answer = None              # consider adding more sofisticate simulated observer: answer =  sim_obs(self.new_stim)
        else:
            try:
                answer = read_joystick()
            except:
                raise ValueError('can\'t find joystick')
            
        if answer is not None: 
#            self.sock.sendto(str(answer).encode('utf_8'), self.addr)   # only if bayesopt not good enough (see comments above)
#            print("sending answer", str(answer).encode('utf_8'))       # send answer to matlab   (UDP_IP_ADDRESS, UDP_PORT_NO)
            result = self.opt.tell(self.new_stim, answer)
            with open('exp_data.pkl', 'wb') as f:
                pickle.dump(self.opt, f)                
            with open('exp_result.pkl', 'wb') as f:
                pickle.dump(result, f)

        if not self.is_simulation:
            self.reset(self.core)
        
        return False if answer is None else True 


"""
experiment library. could eventually be saved as an independent module, but realism does not refresh modules.
"""

import numpy as np
import scipy.interpolate as spi
import serial
from pyquaternion import Quaternion


class OBE:
    def __init__(self, start_pos, start_direct, start_Up, start_time, start_T,
                 extraction_duration = 7, height = 1.5, final_pitch = -120, finalOffset = 0, 
                 extractionAngle = 0, vibState_duration = 0, appmove_duration = 1,
                 sound_type = 0, sound_traj_gain = 0):
        
        self.obe_state              = 0          # 0: vibrational state, 1: OBE, 2: end
        self.start_time             = start_time
        self.next_vib               = 0          # [ms]  start with a vibration, then reset. avoids sending too many commands         

        self.the_self               = Entity(start_pos, start_direct, start_Up, start_T)
        self.the_body               = Entity(start_pos, start_direct, start_Up, start_T)
        self.sound_source           = Sound_source(start_pos, sound_type)
        
        self.vibState_duration      = vibState_duration
        self.appmove_duration       = appmove_duration        

        self.the_self.traj.compute_traj(extraction_duration, height, final_pitch,
                                         finalOffset, extractionAngle, self.the_body)
        self.sound_source.traj.compute_traj(extraction_duration, sound_traj_gain * height, 0.,   # final pitch should not matter, i am using here 0,0,0
                                         sound_traj_gain * finalOffset, extractionAngle, self.the_body)

    
    def update(self, time):        
        t = time - self.start_time
        
        # if in vibrational state, trigger vibrators
        if t < self.vibState_duration and self.vibState_duration > 0:                
            if t > self.next_vib:
                self.the_body.vibrate(t, self.vibState_duration)
                self.next_vib += .5     # sec. it means next command is sent after 0.5 sec          
        else:
            self.the_body.vib_device.appmove_SoA(self.appmove_duration)         # play apparent motion
            self.obe_state = 1      
        
        # start OBE stimulus                
        if self.obe_state == 1:
            t1 = time - self.vibState_duration - self.start_time
            self.the_self.move(t1, self.the_body)
            self.sound_source.move(t1)
            if t > (self.the_self.traj.extraction_duration + self.vibState_duration):
                self.obe_state = 2

    def animate(self, pos, LAP, UP, sound_pos):
        import matplotlib.pyplot as plt

        UP  = np.array(UP)  + np.array(pos)
        
        ax.plot([pos[0], LAP[0]], [pos[2], LAP[2]], [pos[1], LAP[1]], color = 'red')
        ax.plot([pos[0], UP[0]], [pos[2], UP[2]], [pos[1], UP[1]], color = 'green')
        ax.scatter(pos[0], pos[2], pos[1], color = 'black')
        ax.scatter(sound_pos[0], sound_pos[2], sound_pos[1], color = 'cyan')
        plt.show()
    

class Entity():    
    def __init__(self, pos, direct, UP, T):
        self.pos            = np.array(pos)
        self.direct         = np.array(direct)
        self.UP             = np.array(UP)
        self.LAP            = pos[0] + direct[0], pos[1] + direct[1], pos[2] + direct[2]  # look-at-point
        self.T              = T
        self.T0             = T
        
        self.traj           = Trajectory()
        self.vib_device     = Vibrators()
        self.vib_intensity  = 1
        
    def move(self, t, body):
        new_pos       = np.array([self.traj.x(t), self.traj.y(t), self.traj.z(t)]) + self.T0[0:3, 3]
        new_rot       = np.array([self.traj.p(t), self.traj.q(t), self.traj.r(t)])  # [DOUBLE CHECK] original camera orientation should always be 0, right?

        do_yaw        = Quaternion(axis = [0,1,0], angle = new_rot[2]*np.pi/180) 
        right_v       = np.cross(do_yaw.rotate(body.direct), self.UP)               # copute new right vector
        do_pitch      = Quaternion(axis = right_v, angle = new_rot[1]*np.pi/180)    # rotate areound right vector (=pitch in head coord)

        # compute rotation and transformation matrices
        R             = np.matmul(do_pitch.rotation_matrix, do_yaw.rotation_matrix)
        self.T        = np.vstack((np.hstack((R, np.array([[new_pos[0]], [new_pos[1]], [new_pos[2]]]))),[0,0,0,1]))
        
        # should be superfluous
        self.UP       = np.matmul(self.T[0:3, 0:3], body.UP)
        self.direct   = np.matmul(self.T[0:3, 0:3], body.direct)
        self.pos      = self.T[0:3, 3]
        self.LAP      = self.pos + self.direct 

    def vibrate(self, t, tot_duration):
        amplitude = np.ceil((t/tot_duration)**2 * 8) + 1
        self.vib_device.vibrate(10, amplitude)            # magic 10 is for duraiton, change it if crashes        
          
        
class Trajectory():    
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.p = []
        self.q = []
        self.r = []
        self.extraction_duration = -1;
       
    def interpolate_traj(self, t_i, final, node, scale = 1):        
        y = spi.CubicSpline(t_i, np.array([final * node[0], 
                                           final * node[1], 
                                           final * node[2],  
                                           final * node[3]]) * scale)
        #y = spi.CubicSpline(t_i, final * np.array([node]).T * scale )  # same??  currently not...
        return y

    def compute_traj(self, extraction_duration, height, final_pitch, finalOffset, extractionAngle, body):
        # first, a little trick to guarantee movements is always backward: take the opposite of view direction (which is normalized),
        # then rotate it by extractionAngle, and preserve the horizontal components. then, multiply by offset and add height.
        # ([todo] consider normalizing again after removing vertical component...))
        do_yaw     = Quaternion(axis = [0,1,0], angle = extractionAngle*np.pi/180)
        final_pos  = do_yaw.rotate(-np.array(body.direct))                                        

        final_coord = np.array([final_pos[0] * finalOffset, height, final_pos[2] * finalOffset,   # positions
                                0, final_pitch,  extractionAngle])                                # rotations
        
        t_i         = np.array([0,  .3,  .7,  1]) * extraction_duration
        self.x      = self.interpolate_traj(t_i, final_coord[0], [0,  .2 ,  .8,  1])
        self.y      = self.interpolate_traj(t_i, final_coord[1], [0,  .6 ,  .9,  1])
        self.z      = self.interpolate_traj(t_i, final_coord[2], [0,  .2 ,  .8,  1])
        self.p      = self.interpolate_traj(t_i, final_coord[3], [0,  .05,  .7,  1])
        self.q      = self.interpolate_traj(t_i, final_coord[4], [0,  .05,  .7,  1])
        self.r      = self.interpolate_traj(t_i, final_coord[5], [0,  .05,  .7,  1])
        
        self.extraction_duration = extraction_duration

    
class Sound_source():
    def __init__(self, pos, sound_type):
        self.sound_type  = sound_type
        self.pos         = np.array(pos)
        self.pos0        = np.array(pos)
        self.traj        = Trajectory()
        
    def move(self, t):
        self.pos   = np.array([self.traj.x(t), self.traj.y(t), self.traj.z(t)]) + self.pos0   

    
class Vibrators():
    def __init__(self):
        try:
            self.ser = serial.Serial('COM8', 38400, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, 
                                      stopbits=serial.STOPBITS_ONE, timeout=None, xonxoff=False, rtscts=False, 
                                      write_timeout=None, dsrdtr=False, inter_byte_timeout=None, exclusive=None)
            self.ser.open()
        except:
            pass
   
    def vibrate(self, duration, amplitude):                        
        active  = np.arange(0,12) # all
        SoA     = int(0) # 0 = simultaneous activation.  check that soa dos are in ms
        DoS     = int(duration)
        self.command_stimulus(SoA, DoS, active, amplitude)
    
    def appmove_SoA(self, appmove_duration):
        # play one apparent motion to kickstart the obe
                N = 6                                       # number of active vibrators                         
                R = .2                                      # SoA/DoS ratio. rule of thumb: SoA = .5 DoS
                DoS = int(appmove_duration*1000/(1 + R*(N-1)))   # based on the formula T = DoS - SoA * (n-1)  ;  ms (integers!) required by the protocol
                SoA = int(R * DoS)
                sequence1 = np.array([6,7,8,9,10,11,-1,-1,-1,-1,-1,-1])
                sequence2 = np.array([0,1,2,3,4,5,-1,-1,-1,-1,-1,-1])
                self.command_stimulus(SoA, DoS, sequence1, 9)
                self.command_stimulus(SoA, DoS, sequence2, 9) # not sure this is good for triggering both displays simultaneously
    
    def command_stimulus(self, SoA, DoS, activation_sequence, vib_int):
        low_SoA = SoA & 127
        high_SoA = (SoA >> 7) & 127
        low_DoS = DoS & 127
        high_DoS = (DoS >> 7) & 127
        
        try:
            self.ser.write(bytearray([255]))
            for vib in activation_sequence:
                val = vib*10 + vib_int if vib != -1 else 0
                self.ser.write(bytearray([val]))
            self.ser.write(bytearray([high_SoA,low_SoA,high_DoS,low_DoS,170]))
            
        except:
            pass
        
""" end of library """


if __name__ == "__main__":          # simulate and plot the obe traj
    with Experiment(simulation = 1) as e1:
        for t in np.arange(0,7.1, 10/60):
            e1.update(t)
        
        try:            # for debugging using camera view, remove in future versions
            import os
            import time
            os.rename('view_matrices.csv', 'view_matrices_' + time.strftime("%Y%m%d_%H%M%S") + '.csv')
        except:
            pass