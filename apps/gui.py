from LoG.render.renderer import NaiveRendererAndLoss
from LoG.utils.easyvolcap_utils import Viewer
from easyvolcap.engine import args
import torch
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files
from easyvolcap.utils.console_utils import catch_throw
import numpy as np
from plyfile import PlyElement, PlyData
import torch

def write_gaussians_to_ply(ret, filename='output.ply'):
    # Get data as numpy arrays
    xyz = ret["xyz"].cpu()
    colors = ret["colors"].cpu()
    opacity = ret["opacity"].cpu().squeeze()
    scaling = ret["scaling"].cpu()
    rotation = ret["rotation"].cpu()
    shs = ret["shs"].cpu()

    # Build PLY data array with dummy normals
    num_points = xyz.shape[0]
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # Dummy normal values
        # ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'),
        # ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'),
        # ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'),
        # ('f_rest_9', 'f4'), ('f_rest_10', 'f4'), ('f_rest_11', 'f4'),
        # ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'),
        # ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'),
        # ('f_rest_18', 'f4'), ('f_rest_19', 'f4'), ('f_rest_20', 'f4'),
        # ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'),
        # ('f_rest_24', 'f4'), ('f_rest_25', 'f4'), ('f_rest_26', 'f4'),
        # ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'),
        # ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'),
        # ('f_rest_33', 'f4'), ('f_rest_34', 'f4'), ('f_rest_35', 'f4'),
        # ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'),
        # ('f_rest_39', 'f4'), ('f_rest_40', 'f4'), ('f_rest_41', 'f4'),
        # ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    print(xyz.shape, colors.shape, opacity.shape, scaling.shape, rotation.shape, shs.shape)

    # xyz = xyz[:]
    # colors = colors[num_points_original - num_points:num_points_original]
    # opacity = opacity[num_points_original - num_points:num_points_original]
    # scaling = scaling[num_points_original - num_points:num_points_original]
    # rotation = rotation[num_points_original - num_points:num_points_original]

    ply_data = np.empty(num_points, dtype=dtype)

    # Assign values
    ply_data['x'] = xyz[:, 0]
    ply_data['y'] = xyz[:, 1]
    ply_data['z'] = xyz[:, 2]

    # Dummy normals (e.g., all pointing up)
    ply_data['nx'] = np.zeros(num_points, dtype=np.float32)
    ply_data['ny'] = np.zeros(num_points, dtype=np.float32)
    ply_data['nz'] = np.ones(num_points, dtype=np.float32)



    ply_data['f_dc_0'] = colors[:, 0]
    ply_data['f_dc_1'] = colors[:, 1]
    ply_data['f_dc_2'] = colors[:, 2]

    
    # # # Also fill in dummy data for f_rest_0 to f_rest_44
    # ply_data['f_rest_0'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_1'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_2'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_3'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_4'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_5'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_6'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_7'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_8'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_9'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_10'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_11'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_12'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_13'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_14'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_15'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_16'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_17'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_18'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_19'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_20'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_21'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_22'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_23'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_24'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_25'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_26'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_27'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_28'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_29'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_30'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_31'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_32'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_33'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_34'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_35'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_36'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_37'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_38'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_39'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_40'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_41'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_42'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_43'] = np.zeros(num_points, dtype=np.float32)
    # ply_data['f_rest_44'] = np.zeros(num_points, dtype=np.float32)

    ply_data['opacity'] = opacity

    ply_data['scale_0'] = scaling[:, 0]
    ply_data['scale_1'] = scaling[:, 1]
    ply_data['scale_2'] = scaling[:, 2]

    ply_data['rot_0'] = rotation[:, 0]
    ply_data['rot_1'] = rotation[:, 1]
    ply_data['rot_2'] = rotation[:, 2]
    ply_data['rot_3'] = rotation[:, 3]

    # Create and write PLY
    el = PlyElement.describe(ply_data, 'vertex')
    PlyData([el], text=False).write(filename)
    print(f"PLY file with dummy normals written to: {filename}")

def load_gs():
    torch.random.manual_seed(0)
    device = torch.device('cuda:0')
    # renderer
    renderer = NaiveRendererAndLoss(use_origin_render=False, background = [1., 1., 1.])
    renderer.to(device)
    filename = args.opts['filename']
    ckptname = args.opts['ckptname']
    model_cfg = Config.load(filename)
    model_cfg = update_global_variable(model_cfg, model_cfg)
    model = load_object(model_cfg.model.module, model_cfg.model.args)
    print('Load model from: ', ckptname)
    state_dict = torch.load(ckptname, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.set_state(enable_sh=True)
    model.to(device)
    model.eval()
    # for key, value in vars(model.gaussian).items():
    #     print(f"{key}: {value}")

    # keys: ['scaling', 'colors', 'xyz', 'opacity', 'rotation', 'shs']

    ret = model.get_all_custom()

    print(ret)
    write_gaussians_to_ply(ret, 'output.ply')


    # print(model.get_all())


    return model, renderer, None, ret

if __name__ == '__main__':
    string =  '{"H":1080,"W":1920,"K":[[2139.83251953125,0.0,960.0],[0.0,2139.83251953125,496.2210388183594],[0.0,0.0,1.0]],"R":[[0.3830258846282959,0.9237375855445862,0.0],[-0.617131233215332,0.25589218735694885,0.7440917491912842],[0.687345564365387,-0.28500640392303467,0.6680806875228882]],"T":[[1.5715787410736084],[-3.9151883125305176],[5.792508125305176]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'
    viewer = Viewer(camera_cfg={'type':'Camera', 'string': string})
    viewer.model, viewer.renderer, viewer.dataloader, ret = load_gs()
    catch_throw(viewer.run)()
