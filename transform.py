import tensorflow.compat.v1 as tf
import math
from utils import *
from depth_utils import *
import cv2


class compose(object):
    '''
    Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
         transforms.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
         ])
    '''

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        for t in self.transform:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transform:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def projective_inverse_warp(img, depth, pose, intrinsics):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 6], in the
            order of tx, ty, tz, rx, ry, rz
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    """
    基于投影将源图像反向扭曲到目标图像平面。

       参数：
         img：源图像 [batch, height_s, width_s, 3]
         depth：目标图像的深度图 [batch, height_t, width_t]
         姿势：目标到源相机的变换矩阵[batch, 6]，顺序为tx, ty, tz, rx, ry, rz
         内在函数：相机内在函数 [batch, 3, 3]
       返回：
         源图像反向扭曲到目标图像平面 [batch, height_t,宽度_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0] / (2 ** s)
        fy = intrinsics[:, 1, 1] / (2 ** s)
        cx = intrinsics[:, 0, 2] / (2 ** s)
        cy = intrinsics[:, 1, 2] / (2 ** s)
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

def velo2depth(calib_name, velo_file_name, height, width):
    with open(f'./calib/{calib_name}.txt', 'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    #R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    #R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    #R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = load_velodyne_points(velo_file_name)
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2 * Tr_velo_to_cam * velo
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    IMG_H = height
    IMG_W = width
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    # filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    # generate color map from depth
    #u, v, z = cam
    #plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    #plt.savefig('depth_test.png',bbox_inches='tight')
    return cam

def get_intrinsics(name):
    with open(f'./calib/{name}.txt', 'r') as f:
        calib = f.readlines()
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    a = [0,1,2]
    P2 = P2[:,a]
    return P2

def tgt_img(src_img, velo_file_name = 'E:\pycharm_project\SfMLearner-master\dyne\dyne_000107.bin',
            calib_name = 'calib',pose_name = '00', src_number = 0, step = 1):
    src_number = src_number + 1
    tgt_number = src_number + step
    height = src_img.shape[0]
    width = src_img.shape[1]
    src_img = tf.reshape(src_img, [1, height, width, 3])

    intrinsics = get_intrinsics(calib_name)
    intrinsics = np.float32(intrinsics)
    intrinsics = tf.reshape(intrinsics, [1,3,3])

    pose = np.loadtxt(open(f'./poses/{pose_name}.txt', "rb"), delimiter=" ")
    pose_src = pose[src_number,:].reshape(3, 4)
    pose_tgt = pose[tgt_number,:].reshape(3, 4)
    pose = compute_pose(pose_tgt, pose_src)
    pose = np.float32(pose).reshape((1,6))
    print(pose)

    pose = tf.reshape(pose, [1,6])
    cam = velo2depth(calib_name, velo_file_name, height, width)
    len = cam.shape[1]
    depth_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            depth_map[i][j] = 10

    for i in range(len):
        x = math.floor(cam[0,i])
        y = math.floor(cam[1,i])
        depth_map[y][x] = cam[2,i]
    '''
    depth_map_show = depth_map
    for i in range(height):
        for j in range(width):
            if depth_map_show[i][j] <= 10:
                depth_map_show[i][j] = 255 - depth_map[i][j]
    '''
    org_shape = dst_shape = depth_map.shape
    depth_map = cv2.dilate(depth_map, (2,2))
    depth_map = bilinear(depth_map, org_shape, dst_shape)
    cv2.imwrite("depth_test.png", depth_map)
    depth_map = np.float32(depth_map)
    depth_map= tf.reshape(depth_map, [1, height, width])
    tgt_img = projective_inverse_warp(src_img, depth_map, pose, intrinsics)
    tgt_img = tf.reshape(tgt_img, [height, width, 3])
    return tgt_img


def make_intrinsics_matrix(self, fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def bilinear(org_img, org_shape, dst_shape):
    dst_img = np.zeros((dst_shape[0], dst_shape[1]))
    dst_h, dst_w = dst_shape
    org_h, org_w = org_shape
    for i in range(dst_h):
        for j in range(dst_w):
            src_x = j * float(org_w / dst_w)
            src_y = i * float(org_h / dst_h)
            src_x_int = j * org_w // dst_w
            src_y_int = i * org_h // dst_h
            a = src_x - src_x_int
            b = src_y - src_y_int

            if src_x_int+1 == org_w or src_y_int+1 == org_h:
                dst_img[i, j] = org_img[src_y_int, src_x_int]
                continue
            # print(src_x_int, src_y_int)
            dst_img[i, j] = (1. - a) * (1. - b) * org_img[src_y_int+1, src_x_int+1] + \
                            (1. - a) * b * org_img[src_y_int, src_x_int+1] + \
                            a * (1. - b) * org_img[src_y_int+1, src_x_int] + \
                            a * b * org_img[src_y_int, src_x_int]
    return dst_img

def compute_pose(tgt_pose, src_pose):
    filler = np.array([0, 0, 0, 1]).reshape((1,4))
    tgt_pose = tgt_pose.astype(np.float32).reshape(3,4)
    tgt_pose = np.concatenate((tgt_pose, filler), axis=0)
    src_pose = src_pose.astype(np.float32).reshape(3,4)
    src_pose = np.concatenate((src_pose, filler), axis=0)
    rel_pose = np.dot(np.linalg.inv(src_pose), tgt_pose)
    rel_6DOF = getPose_fromT(rel_pose)
    return rel_6DOF

def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

def getPose_fromT(T):
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]
    rz = math.atan2(T[2,1], T[2,2])
    ry = math.atan2(-T[2,0], math.sqrt(T[2,1]*T[2,1] + T[2,2]*T[2,2]))
    rx = math.atan2(T[1,0], T[0,0])
    pose = np.array([x, y, z, rx, ry, rz])

    return pose

