"""Tool to convert Waymo Open Dataset to pickle files.
    Adapted from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder 
import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 
import numpy as np
import pandas as pd
import math as m
import re

tf.enable_v2_behavior()

fnames = None 
LIDAR_PATH = None
ANNO_PATH = None 
NUM_PEOPLE = 7


def rot_z(angle):
    '''
    this function gives 3D rotation matrix around z axis
    '''
    return np.array([[m.cos(angle), -m.sin(angle), 0], 
                     [m.sin(angle), m.cos(angle), 0], 
                     [0, 0, 1]])

def get_angle(p0, p1, p2):
    '''
    this function computes angle of p0p1p2 corner
    '''
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    
    return angle

def convert(idx):
    global fnames
    fname = fnames[idx]
    direction = np.random.choice(('fwd', 'bckwrd', 'left', 'right', 'standing'), size=NUM_PEOPLE)
    zone = np.random.choice((0, 1, 2), p = [0.6, 0.25, 0.15], size=NUM_PEOPLE)
    ranges = {0 : [0, 6],
              1 : [6, 12],
              2 : [12, 18]}

    begs_ends = np.array([ranges[i] for i in zone])
    start_dist = np.zeros((3, NUM_PEOPLE))
#    offset_total = np.zeros((3, NUM_PEOPLE))
#    seq_len = np.random.uniform(10, 30, NUM_PEOPLE)
#    seq_beg_time = np.random.uniform(0, 160, NUM_PEOPLE)
    
    for i in range(len(begs_ends)):
        start_dist[:, i] = np.append(np.random.uniform(begs_ends[i, 0], begs_ends[i, 1], 2), 0)    

    df = pd.read_csv('/home/SUOD/SUOD_annos_train_modified.csv')
    dataset = tf.data.TFRecordDataset(fname, compression_type='')

    for frame_id, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = waymo_decoder.decode_frame(frame, frame_id)
        decoded_annos = waymo_decoder.decode_annos(frame, frame_id)
        
        if frame_id == 0:
            timestamp_old = int(decoded_frame['frame_name'].split("_")[-1])
        else:
            timestamp_old = timestamp_new

        timestamp_new = int(decoded_frame['frame_name'].split("_")[-1])

        for i in range(NUM_PEOPLE):
#            if frame_id >= seq_beg_time[i] and frame_id <= seq_beg_time[i] + seq_len[i]:
                df_dir = df.loc[df['direction'] == direction[i]]

                if direction[i] == 'standing':
                    speed_x = 0
                    speed_y = 0
                else:
                    speed_x = np.random.normal(0, 1, 1)[0] 
                    speed_y = np.random.normal(0, 1, 1)[0]
                    speed_total = abs(np.random.normal(3.5, 1, 1)[0])
                    #in order to make sum of sqares equal to speed_total
                    norm = m.sqrt(speed_x**2 + speed_y**2)
                    speed_x = speed_x * m.sqrt(speed_total)  * 1000 / (60 * 60 * norm)
                    speed_y = speed_y * m.sqrt(speed_total)  * 1000 / (60 * 60 * norm)

                #offset = np.append(np.array((speed_x, speed_y)) * (timestamp_new - timestamp_old) * 1e-6, 0)
                #offset_total[:, i] = offset
                sample_human = df_dir.sample(1)
                points = np.load(sample_human['points'].values[0])
                bbox_position = np.load(sample_human['bbox vertices'].values[0])

                cx = sample_human['centers_x'].values[0]
                cy = sample_human['centers_y'].values[0]
                cz = sample_human['centers_z'].values[0]
                cent = np.array((cx, cy, cz))        
                face_plane = [int(i) for i in sample_human['face'].values[0][1:-1].split(', ')]
                intensity = np.array([int(j)  for i in sample_human['intensity'].values[0][1:-1].split('\n') for j in i.split()])
                intensity = intensity/max(intensity)
                center_fp = points[face_plane].mean(0)
                wlh = np.array([np.double(i) for i in sample_human['wlh'].values[0][1:-1].split()])
                direction_vec = bbox_position[np.array([int(i) for i in sample_human['direction_vec'].values[0][1:-1].split(', ')])].mean(axis=1)
                rotation_from_positive = get_angle(direction_vec[:2], np.array((0, 0)), np.array((0, 1)))

                #compute angle of rotation for human to be facing proper direction
                human_rot_angle = get_angle(np.array([0, 0]), cent[:2], center_fp[:2])
                points_rotated = (points - cent).dot(rot_z(human_rot_angle))
                start_dist[-1, i] = max(0, abs(points_rotated[:, 2].min()))
                human_position = points_rotated  + start_dist[:, i]
                bbox_position = (bbox_position - cent).dot(rot_z(human_rot_angle)) + start_dist[:, i]
                decoded_annos['objects'].append({'id': len(decoded_annos['objects'])+1,
                                                 'name': 'leon_'+str(0),
                                                 'label': 1,
                                                 'box': np.hstack((cent, wlh, np.array((rotation_from_positive, 0, 0)))),
                                                 #x, y, z, length, width, height, rotation from positive x axis clockwisely
                                                 'num_points': len(human_position),
                                                 'detection_difficulty_level': 0,
                                                 'combined_difficulty_level': 0,
                                                 'global_speed': np.array([speed_x, speed_y]),
                                                 'global_accel': np.array([0, 0])})
                
                elongation = np.full((len(intensity), 1), sum(start_dist[:, i]**2)*0.002/30)
                elongation += np.random.normal(0, elongation[0]*0.05, (len(intensity), 1))

                decoded_frame['lidars']['points_xyz'] = np.vstack((decoded_frame['lidars']['points_xyz'], human_position))
                decoded_frame['lidars']['points_feature'] = np.vstack((decoded_frame['lidars']['points_feature'], np.hstack((intensity.reshape(-1, 1), elongation))))        
        
        with open(os.path.join(LIDAR_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_frame, f)
        
        with open(os.path.join(ANNO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_annos, f)


def main(args):
    global fnames 
    fnames = list(glob.glob(args.record_path))

    print("Number of files {}".format(len(fnames)))

    with Pool(128) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    ANNO_PATH = os.path.join(args.root_path, 'annos')

    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    if not os.path.isdir(ANNO_PATH):
        os.mkdir(ANNO_PATH)
    
    main(args)
