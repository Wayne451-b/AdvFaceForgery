import numpy as np
import torch
from config import device
import cv2
import numpy as np
import cv2
import torch
import kornia
import os


def ESDR709_to_LSDR2020(ERGB709, dim):
    LRGB = ERGB709 ** 2.4
    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1
    LR2020 = 0.6274 * LR + 0.3293 * LG + 0.0433 * LB
    LG2020 = 0.0691 * LR + 0.9195 * LG + 0.0114 * LB
    LB2020 = 0.0164 * LR + 0.0880 * LG + 0.8956 * LB
    LRGB2020 = torch.cat([LR2020, LG2020, LB2020], dim=dim)  # hw3
    return LRGB2020 * 100

def EOTF_PQ_cuda(ERGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ERGB = torch.clamp(ERGB, min=1e-10, max=1)

    X1 = ERGB ** (1 / m2)
    X2 = X1 - c1
    X2[X2 < 0] = 0

    X3 = c2 - c3 * X1

    X4 = (X2 / X3) ** (1 / m1)
    return X4 * 10000


def EOTF_PQ_cuda_inverse(LRGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    RGB_l = LRGB / 10000
    RGB_l = torch.clamp(RGB_l, min=1e-10, max=1)

    X1 = c1 + c2 * RGB_l ** m1
    X2 = 1 + c3 * RGB_l ** m1
    X3 = (X1 / X2) ** m2
    return X3


def SDR_to_ICTCP(ERGB, dim=-1):
    ERGB = torch.from_numpy(ERGB[:, :, ::-1] / 255)
    LRGB = ESDR709_to_LSDR2020(ERGB, dim=dim)

    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1
    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=dim)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=dim)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I, T, P], dim=dim)  # hw3
    return ITP


def ICTCP_to_SDR(ITP, dim=-1):
    input_is_numpy = isinstance(ITP, np.ndarray)
    if input_is_numpy:
        ITP = torch.from_numpy(ITP.astype(np.float32)) 
    else:
        ITP = ITP.float() 

    if torch.cuda.is_available():
        ITP = ITP.cuda()

    M1 = torch.tensor([
        [2048, 2048, 0],
        [6610, -13613, 7003],
        [17933, -17390, -543]
    ], dtype=torch.float32, device=ITP.device) / 4096.0  

    M1_inv = torch.linalg.inv(M1)

    I, T, P = torch.split(ITP, 1, dim=dim)
    ITP_vec = torch.cat([I, T, P], dim=dim)

    orig_shape = ITP_vec.shape
    ITP_vec = ITP_vec.view(-1, 3)

    if ITP_vec.dtype != M1_inv.dtype:
        ITP_vec = ITP_vec.type(M1_inv.dtype)

    ELMS_vec = torch.mm(ITP_vec, M1_inv.T)
    ELMS = ELMS_vec.view(orig_shape)

    LMS = EOTF_PQ_cuda(ELMS)
    M2 = torch.tensor([
        [1688, 2146, 262],
        [683, 2951, 462],
        [99, 309, 3688]
    ], dtype=torch.float32, device=LMS.device) / 4096.0
    M2_inv = torch.linalg.inv(M2)

    L, M, S = torch.split(LMS, 1, dim=dim)
    LMS_vec = torch.cat([L, M, S], dim=dim)
    LMS_vec_flat = LMS_vec.view(-1, 3)

    if LMS_vec_flat.dtype != M2_inv.dtype:
        LMS_vec_flat = LMS_vec_flat.type(M2_inv.dtype)

    LRGB2020_vec = torch.mm(LMS_vec_flat, M2_inv.T)
    LRGB2020 = LRGB2020_vec.view(orig_shape)

    LRGB2020 = LRGB2020 / 100.0

    M3 = torch.tensor([
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956]
    ], dtype=torch.float32, device=LRGB2020.device)
    M3_inv = torch.linalg.inv(M3)

    LRGB2020_flat = LRGB2020.view(-1, 3)

    if LRGB2020_flat.dtype != M3_inv.dtype:
        LRGB2020_flat = LRGB2020_flat.type(M3_inv.dtype)

    LRGB709_vec = torch.mm(LRGB2020_flat, M3_inv.T)
    LRGB709 = LRGB709_vec.view(orig_shape)
    ERGB709 = torch.clamp(LRGB709, 1e-7, 1.0) ** (1.0 / 2.4)

    ERGB709 = ERGB709.cpu().numpy()
    ERGB709 = (np.clip(ERGB709, 0, 1) * 255).astype(np.uint8)
    ERGB709 = cv2.cvtColor(ERGB709, cv2.COLOR_RGB2BGR)

    return ERGB709


def rgb2ycbcr_np(img):
    #image as np.float, within range [0,255]

    A = np.array([[0.2568, 0.5041, 0.0979], [-0.1482, -0.2910, 0.4392], [0.4392, -0.3678, -0.0714]])
    ycbcr = img.dot(A.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr[:,:,0] += 16
    return ycbcr

def ycbcr_to_tensor(img_ycc):
    img_ycc = img_ycc.transpose(2,0,1) / 255.
    img_ycc_tensor = torch.Tensor(img_ycc)
    return img_ycc_tensor.unsqueeze(0)

def ycbcr2rgb_np(img):
    invA = np.array([[1.1644, 0.0, 1.5960], [1.1644, -0.3918, -0.8130], [1.1644, 2.0172, 0.0] ])
    img = img.astype(np.float32)
    img[:,:,[1,2]] -= 128
    img[:,:,0] -= 16
    rgb = img.dot(invA.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.around(rgb)

def ycbcr_to_rgb(img_ycc):
    img_ycc = img_ycc.squeeze(0)
    img_ycc = img_ycc.permute(1,2,0).contiguous().view(-1,3).float()
    invA = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.0172],
                        [1.5960, -0.8130, 0]])

    invb = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
    invA, invb = invA.to(device), invb.to(device)
    invA.requires_grad = False
    invb.requires_grad = False
    img_ycc = (img_ycc + invb).mm(invA)
    img_ycc = img_ycc.view(256, 256, 3)
    img_ycc = img_ycc.permute(2,0,1)
    img_ycc = img_ycc.unsqueeze(0)
    img_ycc = torch.clamp(img_ycc, min=0., max=1.)
    return img_ycc
