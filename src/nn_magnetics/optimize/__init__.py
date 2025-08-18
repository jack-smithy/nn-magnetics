# from nn_magnetics.models.networks import AngleAmpCorrectionNetwork
# from nn_magnetics.data.create_data import generate_points_grid
# from magpylib import magnet
# from magpylib_material_response import meshing, demag
# import torch

# MODEL_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_v2/2025-04-24 14:35:47.849107/best_weights.pt"
# HIDDEN_DIM_FACTOR = 6
# DIMENSIONS = (1, 1, 1)
# SUSCEPTIBILITIES = (1, 1, 1)
# POLARIZATION = (0, 0, 1)
# OBSERVERS = generate_points_grid(26, DIMENSIONS[0], DIMENSIONS[1])

# model = AngleAmpCorrectionNetwork.load_from_path(
#     path=MODEL_PATH,
#     activation=torch.nn.functional.silu,
#     save_path=None,
#     save_weights=False,
# ).to(torch.float64)

# model.eval()

# cuboid = magnet.Cuboid(dimension=DIMENSIONS, polarization=POLARIZATION)

# mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
# demag.apply_demag(mesh, SUSCEPTIBILITIES, inplace=True)

# B = mesh.getB(OBSERVERS)

# B_target = torch.from_numpy(B)


# def func(susc):
#     xyz = torch.from_numpy(OBSERVERS)
#     dims_expanded = (
#         torch.tensor(DIMENSIONS)[:2]
#         .repeat((xyz.shape[0], 1))
#         .unsqueeze(0)
#         .expand(xyz.shape[0], -1)
#     )
#     input_tensor = torch.cat([dims_expanded, susc, xyz], dim=1)
#     B_pred = model(input_tensor)
#     return torch.nn.functional.mse_loss(B_pred, B_target)


# dim_preds = []
# for i in range(3):
#     susc = torch.nn.Parameter(torch.rand(3))
#     opt = torch.optim.Adam(params=[susc], lr=0.01)

#     for step in range(51):
#         opt.zero_grad()

#         loss = func(susc)
#         loss.backward()
#         opt.step()

#     dim_preds.append(susc)

# susc_t = torch.stack(dim_preds)
# susc_t_mean = susc_t.mean(0)
# susc_t_std = susc_t.std(0)


# def format_results(susc: torch.Tensor, precision: int = 5) -> str:
#     dim_t_mean = susc.mean(0)
#     dim_t_std = susc.std(0)

#     a_mean, b_mean, c_mean = (
#         round(dim_t_mean[0].item(), precision),
#         round(dim_t_mean[1].item(), precision),
#         round(dim_t_mean[2].item(), precision),
#     )

#     a_std, b_std, c_std = (
#         round(dim_t_std[0].item(), precision),
#         round(dim_t_std[1].item(), precision),
#         round(dim_t_std[2].item(), precision),
#     )

#     return f"chi_x={a_mean}±{a_std}, chi_y={b_mean}±{b_std}, chi_z={c_mean}±{c_std}"


# print(format_results(susc_t))
