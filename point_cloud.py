import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

import utils
from Q1 import q1_a, q1_c
from Q2 import q2
from Q3 import q3

def hw3(question:str, testing:bool = False):
    if question == 'q1_a':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P += 0.02 * np.random.randn(P.shape[0], 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)
        

        normal, center = q1_a(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q1_b':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P = P + 0.02 * np.random.randn(P.shape[0], 3)

        # add outlier noise
        P[np.arange(8)] += np.random.randn(8, 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)

        normal, center = q1_a(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q1_c':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P = P + 0.02 * np.random.randn(P.shape[0], 3)

        # add outlier noise
        P[np.arange(8)] += np.random.randn(8, 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)

        normal, center = q1_c(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q2':
        scene_data = np.load('object3d.npy')
        pc, normals, rgb = np.array_split(scene_data, 3, axis=1)

        # segment region of interest (for testing ONLY)
        if testing:
            roi = np.array(((-np.inf, 0.5), (0.2, 0.4), (0.1, np.inf)))
        else:
            roi = np.array(((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)))
        mask = utils.within_roi(pc, roi)

        center, radius = q2(pc[mask], normals[mask])

        # plot point cloud and fitted sphere
        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(30, -115)
        utils.plot_sphere(ax, center, radius)
        ax.scatter(*pc[::4].T, c=rgb[::4], s=0.04)
        plt.tight_layout()
        plt.show()

    elif question == 'q3':
        scene_data = np.load('object3d.npy')
        pc, normals, rgb = np.array_split(scene_data, 3, axis=1)

        # segment region of interest
        roi = np.array(((0.4, 0.6), (-np.inf, 0.2), (0.1, np.inf)))
        mask = utils.within_roi(pc, roi)

        center, axis, radius = q3(pc[mask], normals[mask])
        axis = axis / np.linalg.norm(axis)

        # plot point cloud and fitted cylinder
        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(30, -115)
        utils.plot_cylinder(ax, center, axis, radius)
        ax.scatter(*pc[::4].T, c=rgb[::4], s=0.04)
        plt.tight_layout()
        plt.show()

    else:
        print('Invalid question: choose from '
              '{q1_a, q1_c, q2, q3, q4_a, q4_b, q4_c}.')


valid_questions = ['q1_a', 'q1_b', 'q1_c', 'q2', 'q3', 'q4_a', 'q4_b', 'q4_c']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter question')
    parser.add_argument('-q','--question', type=str, help='Enter question')
    parser.add_argument('-t','--testing', action='store_true', help='testing mode')
    args = parser.parse_args()

    if args.question == None or args.question.lower() not in valid_questions:
        raise ValueError(f'Error: please enter a valid question as a parameter: {valid_questions}')

    hw3(args.question, args.testing)