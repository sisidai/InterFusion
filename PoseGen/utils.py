import torch
import numpy as np
from torch.nn import functional as F
import clip
import smplx
from render import render_single_batch


device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" # the neural render does not support inference on cpu


clip_model, _ = clip.load('ViT-B/32', device)
clip_model.eval()


smpl_path = '../data/smplx_model'
smpl = smplx.create(smpl_path, 'smplx', flat_hand_mean=True).to(device)


def pose_padding(pose):
    assert pose.shape[-1] == 69 or pose.shape[-1] == 63
    if pose.shape[-1] == 63:
        padded_zeros = torch.zeros_like(pose)[..., :6]
        pose = torch.cat((pose, padded_zeros), dim=-1)
    return pose


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(
            1, 3, 1
        ).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(rotation_matrix.shape)
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(rotation_matrix.shape)
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2]
        ], -1
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1]
        ], -1
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2
        ], -1
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0]
        ], -1
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0 + t1_rep * mask_c1 +    # noqa
        t2_rep * mask_c2 + t3_rep * mask_c3
    )    # noqa
    q *= 0.5
    return q


def rot6d_to_rotation_matrix(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def load_pose(pose_path):
    data = np.load(pose_path, allow_pickle=True).item()
    pose = rotation_matrix_to_angle_axis(rot6d_to_rotation_matrix(data['pose'].cpu())).reshape(-1).cpu()

    if pose.shape[-1] == 72:
        pose = pose[..., 3:66]
    # pose = pose_padding(pose)  # torch.Size([69]) # if smpl
    if len(pose.shape) == 1:
        pose = pose.unsqueeze(0)

    return pose


def get_text_feature(text, device):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_feature = text_features[0]
    return text_feature


def get_pose_feature(pose, angles=None):
    ''' SMPL model constructor
                Parameters
                ......
                global_orient: torch.tensor, optional, Bx3
                    The default value for the global orientation variable.
                    (default = None)
                body_pose: torch.tensor, optional, Bx(Body Joints * 3), NUM_BODY_JOINTS == 23
                    The default value for the body pose variable.
                    (default = None)
                ......
    '''
    if len(pose.shape) == 1:
        pose = pose.unsqueeze(0)
    bs = pose.shape[0]

    # adjust the orientation
    global_orient = torch.zeros(bs, 3).type_as(pose)
    global_orient[:, 0] = np.pi / 2

    output = smpl(
        body_pose=pose,
        global_orient=global_orient,
        transl = -smpl().joints[:, 0, :])
    v = output.vertices

    # optional: center the vertices by subtracting their mean :)
    # v = output.vertices - torch.mean(output.vertices, dim=1, keepdim=True)

    f = smpl.faces
    f = torch.from_numpy(f.astype(np.int32)).unsqueeze(0).repeat(bs, 1, 1).to(device)
    if angles is None:
        # optional: the angles can be served as a hyperparameter :)
        angles = (0, 90, 180, 270)
    images = render_single_batch(v, f, angles, device)
    images = F.interpolate(images, size=224)

    # optional: if you want to visualize intermediate rendering results, uncomment the lines below
    # from PIL import Image
    # for idx in range(len(images)):
    #     show_image = np.clip(images[idx].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
    #     save_path = f'./vis'
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     Image.fromarray((show_image * 255).astype(np.uint8)).save(os.path.join(save_path, f'{angles[idx]}.png'))
    # print('[INFO] rendered images saved')

    # use normalized clip
    mean = np.array([0.48145466, 0.4578275, 0.40821073]) # mean_clip
    std = np.array([0.26862954, 0.26130258, 0.27577711]) # std_clip
    images -= torch.from_numpy(mean).reshape(1, 3, 1, 1).to(device)
    images /= torch.from_numpy(std).reshape(1, 3, 1, 1).to(device)
    num_camera = len(angles)
    image_embed = clip_model.encode_image(images).float().view(num_camera, -1, 512)
    return image_embed.mean(0)


def suppress_duplicated_poses(poses, threshold=0.07):
    new_poses = []
    for pose in poses:
        if len(new_poses) == 0:
            new_poses.append(pose)
        else:
        # calculate the minimum distance of the current pose and all poses in new_poses, if the distance is greater than the threshold, add the current pose to new_poses
            min_dis = 10
            for j in range(len(new_poses)):
                cur_dis = torch.abs(pose - new_poses[j]).mean()
                min_dis = min(cur_dis, min_dis)
            if min_dis > threshold:
                new_poses.append(pose)
    poses = torch.stack(new_poses, dim=0)
    return poses


def get_topk_poses(text, data, topk, pre_topk=40):
    with torch.no_grad(): 
        text_feature = get_text_feature(text, device).unsqueeze(0)

        score = F.cosine_similarity(data['codebook_embedding'].to(device), text_feature).view(-1)
        _, indexs = torch.topk(score, pre_topk)
        poses = data['codebook'][indexs].to(device)

        # optional: remove duplicated poses :)
        # poses, indexs = suppress_duplicated_poses(poses)
        
        poses = poses[:topk]
        print(f"The Top{topk} poses are: {indexs[:topk].cpu().numpy().tolist()}")

    return poses


def build_codebook(poses, save_path):
    with torch.no_grad():
        poses = poses.to(device)

        # build your own codebook
        poses_list = []
        pose_features = []

        for i in range(len(poses)):
            pose = poses[i]
            pose_feature = get_pose_feature(pose).squeeze(0)
            poses_list.append(pose)
            pose_features.append(pose_feature)
        pose_embeddings = torch.stack(pose_features)
        poses = torch.stack(poses_list)

        torch.save({'codebook': poses, 'codebook_embedding':pose_embeddings}, save_path)
