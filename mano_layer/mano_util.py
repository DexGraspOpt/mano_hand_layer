import trimesh
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import point_cloud_utils as pcu
import trimesh.sample


Palm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 594, 595, 88, 89, 90, 91, 92, 93, 94,
        95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 123, 126, 127, 128, 129, 130, 131, 132, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148,
        149, 150, 151, 152, 153, 154, 157, 158, 159, 160, 161, 162, 163, 168, 169, 172, 178, 179, 180, 181, 182, 183,
        184, 185, 186, 187, 188, 696, 190, 191, 192, 193, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 214, 215, 216, 217, 218, 219, 220, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 239,
        240, 241, 242, 243, 244, 246, 247, 248, 252, 253, 254, 255, 256, 257, 258, 259, 260, 769, 262, 263, 264, 265,
        266, 770, 268, 269, 270, 271, 775, 776, 274, 275, 276, 277, 278, 279, 284, 285, 286, 288, 289, 290, 291, 292,
        293, 771, 772, 773, 774, 777, 604, 605]

Thumb = [28, 31, 104, 124, 125, 249, 250, 251, 267, 287, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
         708, 709, 710, 711, 712, 713, 714,
         715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
         725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
         738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
         751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
         764, 765, 766, 767, 768]

Thumb_f0 = [28, 31, 104, 124, 125, 249, 250, 251, 267, 287, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
            708, 709, 710, 711, 712, 713, 714,
            715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
            725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
            738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
            751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
            764, 765, 766, 767, 768]
Thumb_f1 = [715, 732, 741, 742, 745, 744, 755, 757, 758, 766, 729, 735, 751, 765, 730, 752, 764, 738, 728, 768,
            727, 767, 743, 747, 720, 748, 717, 750, 734, 761, 737, 724, 762,
            763, 726, 740, 719, 746, 718, 725, 722, 723, 733, 749, 716, 731,
            721, 736, 759, 739, 760, 756]

Index = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 135, 136, 139, 140, 155,
         156, 164, 165, 166, 167, 170, 171, 173, 174, 175, 176, 177, 189, 194, 195, 212,
         213, 221, 222, 223, 224, 225, 226, 237, 238, 245, 261, 272, 273,
         280, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300,
         301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
         312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
         323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
         334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
         346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

Middle = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
          372, 373, 374, 375, 376, 377, 378, 379, 380, 381,
          382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
          395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
          408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
          421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
          434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
          447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
          460, 461, 462, 463, 464, 465, 466, 467]

Ring = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
        482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
        495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
        508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
        521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
        534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
        547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
        560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
        573, 574, 575, 576, 577, 578, 579]

Little = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 596, 597,
          598, 599, 600, 601, 602, 603, 606, 607, 608,
          609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620,
          621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633,
          634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
          647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
          660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
          673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
          686, 687, 688, 689, 690, 691, 692, 693, 694, 695]

Index_f1 = [46, 47, 56, 57, 86, 155, 221, 222, 223, 224, 226, 237, 238, 245, 272, 273, 280, 281, 282, 283] + list(
    range(294, 356))
Index_f2 = list(range(294, 356))

Middle_f1 = [356, 357, 360, 361, 364, 372, 390, 391, 392, 393, 395, 396, 397, 398, 400, 401, 402, 403, 404, 405] + list(
    range(406, 468))
Middle_f2 = list(range(406, 468))

Ring_f1 = [500, 503, 506, 507, 508, 511, 512, 514, 515, 516] + list(range(517, 579))
Ring_f2 = list(range(517, 579))

Little_f1 = [580, 581, 584, 585, 588, 598, 618, 619, 620, 621, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633] + list(
    range(634, 696))
Little_f2 = list(range(634, 696))

def calculate_barycentric_coords_batch(points, vertices):
    """
    Calculate the barycentric coordinates of a batch of points within a set of triangles.

    Args:
        points (numpy.ndarray): A 2D array of 2D or 3D points for which you want to calculate the barycentric coordinates.
        vertices (numpy.ndarray): A 3D array of 2D or 3D vertex coordinates, where the first dimension represents the triangle index,
                                 and the second dimension represents the vertex index (0, 1, 2).

    Returns:
        numpy.ndarray: A 2D array of barycentric coordinates (u, v, w) for each input point.
    """
    a = vertices[:, 0]
    b = vertices[:, 1]
    c = vertices[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    # Compute denominator
    denom = d00 * d11 - d01 * d01

    # Compute barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.stack([u, v, w], axis=1)


def reconstruct_points(vertices, barycentric_coordinates):
    A = vertices[:, 0]
    B = vertices[:, 1]
    C = vertices[:, 2]

    u = barycentric_coordinates[..., 0]
    v = barycentric_coordinates[..., 1]
    w = barycentric_coordinates[..., 2]

    # Reconstruct points
    P = u[:, None] * A + v[:, None] * B + w[:, None] * C
    return P


def sample_visible_points(voxel_size=0.00375, random_colors=None):
    hand = trimesh.load('../assets/hand_zero.obj', force='mesh')
    result = get_surface_point_cloud(hand, scan_count=100, scan_resolution=400)
    points = np.array(result.points)
    points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, points)
    points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, points)
    pc_1 = trimesh.PointCloud(points, colors=(255, 255, 0))

    closest, distance, triangle_id = trimesh.proximity.closest_point(hand, points)

    part_colors = hand.visual.face_colors[triangle_id][:, :3]
    print(part_colors.shape)
    thumb = np.where(part_colors == random_colors[1])[0]
    index = np.where(part_colors == random_colors[2])[0][::3]
    middle = np.where(part_colors == random_colors[3])[0][::3]
    ring = np.where(part_colors == random_colors[4])[0][::3]
    little = np.where(part_colors == random_colors[5])[0][::3]

    np.savetxt('../assets/finger_point_idx/thumb.txt', thumb)
    np.savetxt('../assets/finger_point_idx/index.txt', index)
    np.savetxt('../assets/finger_point_idx/middle.txt', middle)
    np.savetxt('../assets/finger_point_idx/ring.txt', ring)
    np.savetxt('../assets/finger_point_idx/little.txt', little)
    all_finger_idx = np.concatenate([thumb, index, middle, ring, little])

    palm = set(list(range(points.shape[0]))).difference(set(all_finger_idx))
    np.savetxt('../assets/finger_point_idx/palm.txt', np.array(list(palm)))

    pc_2 = trimesh.PointCloud(closest, colors=(255, 0, 255))
    scene = trimesh.Scene([pc_1, pc_2])
    scene.show()
    faces = hand.faces
    verts_idx = faces[triangle_id]
    np.savetxt('../assets/face_idx.txt', triangle_id)
    np.savetxt('../assets/face_vertex_idx.txt', verts_idx)

    triangles = hand.triangles
    triangle_points = triangles[triangle_id]
    coords = calculate_barycentric_coords_batch(closest, triangle_points.copy())
    np.savetxt('../assets/point_coord.txt', coords)

    # reconstruct point
    points = reconstruct_points(triangle_points, coords)
    pc = trimesh.PointCloud(points.squeeze(), colors=(255, 0, 255))
    pc.show()


if __name__ == "__main__":
    import torch
    import numpy as np
    import trimesh
    from manotorch.anchorlayer import AnchorLayer

    from manotorch.manolayer import ManoLayer, MANOOutput

    # all = list(range(0, 778))
    # fingers = Thumb + Index + Middle + Ring + Little
    # palm = set(all).difference(set(fingers))
    # print(palm)
    # exit()

    # initialize layers
    mano_layer = ManoLayer(use_pca=False, center_idx=9)
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    BS = 1
    random_shape = torch.zeros(BS, 10)
    root_pose = torch.tensor([[0, 0, 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    # finger_pose[:, 9:12] = torch.tensor([-2.9802e-08, -7.4506e-09,  1.4932e+00])
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)

    mano_results: MANOOutput = mano_layer(hand_pose, random_shape)
    verts = mano_results.verts
    verts_num = verts.shape[1]
    faces = mano_layer.get_mano_closed_faces()

    vertex_colors = np.array([[0, 0, 0]]).repeat(verts_num, 0)
    random_color = np.random.randint(0, 255, size=(6, 3))
    print(random_color)
    vertex_colors[Palm] = random_color[0]  # Palm
    vertex_colors[Thumb] = random_color[1]  # Thumb
    vertex_colors[Index] = random_color[2]
    vertex_colors[Middle] = random_color[3]
    vertex_colors[Ring] = random_color[4]
    vertex_colors[Little] = random_color[5]

    hand_mesh = trimesh.Trimesh(vertices=verts.squeeze().numpy(), faces=faces.numpy(), vertex_colors=vertex_colors)
    hand_mesh.show()
    hand_mesh.export('../assets/hand_zero.obj')

    sample_visible_points(random_colors=random_color)
