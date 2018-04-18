# accelerate the hdf loading procedure
import h5py
from atari_wrappers import *

@profile
def parse_valset(filename):
    action = "A"
    reward = "R"
    obs = "S"
    terminal = "terminal"
    lives = "lives"

    # parse filename into a list, it could be a comma(,) seperated filename list
    filename = filename.split(",")
    filename = [x.strip() for x in filename if x.strip() != ""]
    print(filename)

    obs_list = []
    action_list = []
    reward_list = []
    terminal_list = []

    for fi in filename:
        print("before file open")
        f1 = h5py.File(fi, 'r')
        print("after file open")        
        _action = list(f1[action])
        _reward = list(f1[reward])
        #import pdb; pdb.set_trace()
	_obs_out = []
        _obs = np.array(f1[obs])
        _terminal = list(f1[terminal])
        assert (len(_action) == len(_reward))
        assert (len(_action) == len(_obs))
        assert (len(_action) == len(_terminal))
        print(len(_obs), '*' * 30)

        for i in range(len(_obs)):
            if True:
                if i < len(_obs) - 1:
                    _obs_out.append( TorcsProcessFrame84.aframe(_obs[i], 120, 160, 'resize'))

        obs_list = obs_list + _obs_out[0:-1]
        action_list = action_list + _action[1:]
        reward_list = reward_list + _reward[1:]
        terminal_list = terminal_list + _terminal[1:]

    return obs_list, reward_list, action_list, terminal_list

a=parse_valset("/data2/hxu/modelRL/demo5/hxu2_torcsSat_Oct__7_13:06:33_PDT_2017.h5")
