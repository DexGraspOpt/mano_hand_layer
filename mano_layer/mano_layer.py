# leap_hand layer for torch
import torch
import trimesh
import os
import numpy as np
import copy
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
import pytorch3d


# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from layer_asset_utils import save_part_mesh, sample_points_on_mesh, sample_visible_points

# All lengths are in mm and rotations in radians


def compute_vertex_normals_wpalfi(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces_packed[:, 0], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 1], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 2], faces_normals)

    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )


class ManoHandLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=True, show_mesh=False, hand_type='right', device='cuda'):
        super().__init__()
        assert to_mano_frame==True
        self.BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
        self.show_mesh = show_mesh
        self.to_mano_frame = to_mano_frame
        self.device = device
        self.name = 'mano_hand'
        self.hand_type = hand_type
        self.finger_num = 5

        self.joints_lower = torch.tensor([-1.57]*45).to(self.device)
        self.joints_upper = torch.tensor([1.57]*45).to(self.device)
        self.joints_mean = (self.joints_lower + self.joints_upper) / 2
        self.joints_range = self.joints_mean - self.joints_lower
        self.n_dofs = 45
        if to_mano_frame:
            self.chain = ManoLayer(use_pca=False, center_idx=9).to(self.device)
        else:
            self.chain = ManoLayer(use_pca=False, center_idx=0).to(self.device)

        self.load_asset()

        # transformation for align the robot hand to mano hand frame
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)

        self.register_buffer('base_2_world', self.to_mano_transform)
        self.zero_root = torch.zeros((1, 3), device=self.device)

        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def load_asset(self):
        assert_dir = os.path.dirname(os.path.realpath(__file__)) + "/../assets/"
        # face vert idx
        face_vert_idx_path = os.path.join(assert_dir, "face_vertex_idx.txt")
        face_vert_idx = np.loadtxt(face_vert_idx_path).astype(np.int32)
        self.face_vert_idx = torch.from_numpy(face_vert_idx).long().unsqueeze(0).to(self.device)
        # face idx
        face_idx_path = os.path.join(assert_dir, "face_idx.txt")
        face_idx = np.loadtxt(face_idx_path).astype(np.int32)
        self.face_idx = torch.from_numpy(face_idx).long().unsqueeze(0).to(self.device)
        # point coord
        point_coord_path = os.path.join(assert_dir, "point_coord.txt")
        point_coord = np.loadtxt(point_coord_path)
        self.point_coord = torch.from_numpy(point_coord).float().unsqueeze(0).to(self.device)

    def get_hand_segment_indices(self):
        assert_dir = os.path.dirname(os.path.realpath(__file__)) + "/../assets/"
        # TODO: hand segment idx for mano hand will be released in the next version
        hand_segment_indices = {}
        hand_finger_indices = {}
        for finger_name in ['palm', 'thumb', 'index', 'middle', 'ring', 'little']:
            idx_path = os.path.join(assert_dir, "finger_point_idx/{}.txt".format(finger_name))
            point_idx = np.loadtxt(idx_path).astype(np.int32)
            hand_finger_indices[finger_name] = torch.from_numpy(point_idx).long()
        return hand_segment_indices, hand_finger_indices

    def recover_points_batch(self, vertices):
        # vertices = TENSOR[NBATCH, 778, 3]
        # face_vert_idx = TENSOR[1, 32, 3]
        # point_coord = TENSOR[1, 32, 3]
        batch_size = vertices.shape[0]
        batch_idx = torch.arange(batch_size)[:, None, None]  # TENSOR[NBATCH, 1, 1]
        indexed_vertices = vertices[batch_idx, self.face_vert_idx, :]  # TENSOR[NBATCH, 32, 3, 3]

        faces_normals = torch.cross(
            indexed_vertices[:, :, 2] - indexed_vertices[:, :, 1],
            indexed_vertices[:, :,  0] - indexed_vertices[:, :, 1],
            dim=2,
        )

        faces_normals = torch.nn.functional.normalize(faces_normals, eps=1e-6, dim=2)

        weights_1 = self.point_coord[:, :, 0:1]  # TENSOR[1, 32, 1]
        weights_2 = self.point_coord[:, :, 1:2]  # TENSOR[1, 32, 1]
        weights_3 = self.point_coord[:, :, 2:3]  # TENSOR[1, 32, 1]
        rebuilt_points = (weights_1 * indexed_vertices[:, :, 0] + weights_2 * indexed_vertices[:, :, 1] +
                          weights_3 * indexed_vertices[:, :, 2])  # TENSOR[NBATCH, 32, 3]
        return rebuilt_points, faces_normals

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 45)): The degrees of freedom of the Robot hand.
       """
        ret = self.chain(theta)
        return ret

    def compute_abnormal_joint_loss(self, theta):
        loss_1 = torch.clamp(theta[:, 0] - theta[:, 4], 0, 1) * 10
        loss_2 = torch.clamp(theta[:, 4] + theta[:, 8], 0, 1) * 10
        loss_3 = -torch.clamp(theta[:, 8] - theta[:, 13], -1, 0) * 10
        loss_4 = torch.abs(theta[:, 12]) * 5
        loss_5 = torch.abs(theta[:, [2, 3, 6, 7, 10, 11,  15, 16]] - self.joints_mean[[2, 3, 6, 7, 10, 11, 15, 16]].unsqueeze(0)).sum(dim=-1) * 2
        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5

    def get_init_angle(self):
        init_angle = torch.tensor([-0.15, 0, 0.6, 0, 0, 0, 0.6, 0, -0.15, 0, 0.6, 0, 0, -0.25, 0, 0.6, 0,
                                        0, 1.2, 0, 0.0, 0], dtype=torch.float, device=self.device)
        return init_angle

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]

        output_verts = (pose[:, :3, :3] @ ret.verts.transpose(1, 2)).transpose(1, 2) + pose[:, :3, 3]
        hand_meshes = []
        for idx in range(bs):
            verts = output_verts[idx]
            faces = self.chain.get_mano_closed_faces()
            vertex_colors = np.array([[0, 255, 255]]).repeat(778, 0)

            hand_mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.numpy(),
                                        vertex_colors=vertex_colors)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        theta_all = torch.cat([self.zero_root.repeat(theta.shape[0], 1), theta], dim=1)
        outputs = self.forward(theta_all)

        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        theta_all = torch.cat([self.zero_root.repeat(theta.shape[0], 1), theta], dim=1)
        outputs = self.forward(theta_all)

        verts = (pose[:, :3, :3] @ outputs.verts.transpose(1, 2)).transpose(1, 2) + pose[:, :3, 3]
        points, normals = self.recover_points_batch(verts)

        return points, normals


class ManoAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            # thumb finger
            1766, 3104, 195, 1848, 213,
            2859, 1460,

            # index finger
            1889, 1261, 749, 1337, 260,  # 2324
            3013,

            # middle finger
            3037, 2049, 2440, 2149, 2145,
            709, 200,

            # ring finger
            1648, 1938, 2788, 906, 1505,
            2070, 2569,  # place holder

            # little finger
            1768, 1884, 764, 1874, 833,  # place holder

            # # plus
            2855, 2965, 576, 2725,  # 2440  2463
            797, 2492, 2990, 2801,
            2490, 2076, 2760, 1282,
            2185, 415,

        ])
        # vert_idx = np.load(os.path.join(self.BASE_DIR, 'anchor_idx.npy'))
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    hand = ManoHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, hand.n_dofs), dtype=np.float32)
    # theta[0, 12:17] = np.array([-0.5, 0.5, 0, 0, 0])
    theta = torch.from_numpy(theta).to(device)
    # theta[:, 9:12] = torch.FloatTensor([-2.9802e-08, -7.4506e-09, 1.4932e+00])
    # limit_middlefinger_right = torch.FloatTensor([-2.9802e-08, -7.4506e-09, 1.4932e+00])  # 9:12
    # print(hand.joints_lower)
    # print(hand.joints_upper)
    # theta = joint_angles_mu = torch.tensor([-0.15, 0, 0.6, 0, -0.15, 0, 0.6, 0, -0.15, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0,
    #                                     0, 1.2, 0, 0.0, 0], dtype=torch.float, device=device)

    # mesh version
    if show_mesh:
        mesh = hand.get_forward_hand_mesh(pose, theta)[0]
        mesh.show()
        mesh.export('../assets/hand_to_mano_frame.obj')
    else:
        # hand_segment_indices, hand_finger_indices = hand.get_hand_segment_indices()
        verts, normals = hand.get_forward_vertices(pose, theta)
        print(verts.shape)

        pc_list = []
        for finger_name in ['palm', 'thumb', 'index', 'middle', 'ring', 'little']:
            finger_indices = hand.hand_finger_indices[finger_name]
            points = verts[:, finger_indices].squeeze().cpu().numpy()
            print(points.shape)
            pc_tmp = trimesh.PointCloud(points, np.random.randint(0, 255, size=(3)))
            pc_list.append(pc_tmp)

        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        mesh = trimesh.load(os.path.join(hand.BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        anchor_layer = ManoAnchor()
        # anchor_layer.pick_points(verts.squeeze().cpu().numpy())
        anchors = anchor_layer(verts).squeeze().cpu().numpy()
        pc_anchors = trimesh.PointCloud(anchors[:46], colors=(255, 0, 255))

        scene = trimesh.Scene([*pc_list, ray_visualize])
        scene.show()