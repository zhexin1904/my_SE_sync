#!/usr/bin/env python
import argparse
import os

import random
import gtsam
import jrl
import numpy as np
from copy import copy
from string import ascii_letters


Rxp = gtsam.Rot3(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
Rxn = gtsam.Rot3(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

Ryp = gtsam.Rot3(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
Ryn = gtsam.Rot3(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))

Rzp = gtsam.Rot3(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
Rzn = gtsam.Rot3(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))


# GLOBALS
ODOM_OPTIONS_GRIDWORLD = [
    gtsam.Pose3(gtsam.Rot3.Identity(), np.array([1, 0, 0])),  # Move forward
    gtsam.Pose3(Rzp, np.array([0, 0, 0])),  # Turn z
    gtsam.Pose3(Rzn, np.array([0, 0, 0])),  # Turn -z
    gtsam.Pose3(Ryp, np.array([0, 0, 0])),  # Turn y
    gtsam.Pose3(Ryn, np.array([0, 0, 0])),  # Turn -y
    gtsam.Pose3(Rxp, np.array([0, 0, 0])),  # Turn x
    gtsam.Pose3(Rxn, np.array([0, 0, 0])),  # Turn -x
]


def handle_args():
    parser = argparse.ArgumentParser(
        description="Generates a random multi-robot grid-world pose graph dataset in jrl format."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="The output directory."
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="The base name for the dataset."
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=1,
        help="The number of datasets to generate.",
    )
    parser.add_argument(
        "-nr",
        "--number_robots",
        type=int,
        default=2,
        help="The number of robots",
    )
    parser.add_argument(
        "-np",
        "--number_poses",
        type=int,
        default=500,
        help="Number of poses for each robot",
    )
    parser.add_argument(
        "--odom_probs",
        type=float,
        nargs="+",
        default=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        help="Odom action probabilities [forward, +-z, +-y, +-x]",
    )

    parser.add_argument(
        "--loop_closure_index_threshold",
        type=int,
        default=5,
        help="Number steps before a prior pose can be identified as a loop closure",
    )
    parser.add_argument(
        "--loop_closure_distance_threshold",
        type=int,
        default=3,
        help="Max distance between current pose and previous pose for a loop closure to be detected",
    )
    parser.add_argument(
        "--loop_closure_probability",
        type=float,
        default=0.8,
        help="Probability that, given a loop closure exists by the threshold criteria, it is detected by the robot",
    )

    parser.add_argument(
        "--comm_range",
        type=float,
        default=20,
        help="Distance threshold for communication between robots",
    )
    parser.add_argument(
        "--comm_freq",
        type=float,
        default=5,
        help="Number of poses between communication",
    )

    parser.add_argument(
        "--prior_noise_sigmas",
        type=float,
        nargs="+",
        default=[1, 1, 1, 10, 10, 10],
        help="Sigmas for diagonal noise model of odometry measurements",
    )

    parser.add_argument(
        "--robot_zero_prior_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.001, 0.001, 0.001, 0.01, 0.01, 0.01],
        help="Sigmas for diagonal noise model of odometry measurements",
    )

    parser.add_argument(
        "--odom_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        help="Sigmas for diagonal noise model of odometry measurements",
    )

    parser.add_argument(
        "--loop_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        help="Sigmas for diagonal noise model of intra-robot loop closure measurements",
    )

    parser.add_argument(
        "--comm_loop_measurement_type",
        type=str,
        default="pose",
        help="The measurement type for inter-robot measurements [pose, range]",
    )

    parser.add_argument(
        "--comm_loop_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        help="Sigmas for diagonal noise model of inter-robot loop closure measurements",
    )

    parser.add_argument(
        "--initialization_type",
        type=str,
        default="odom",
        help="What initialization type to use ('odom', 'gt', 'noisy_gt').",
    )

    parser.add_argument(
        "--initialization_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.2, 0.2, 0.2, 1, 1, 1],
        help="Sigmas to use for 'noisy-gt' initialization.",
    )

    parser.add_argument(
        "--xlims",
        type=float,
        nargs="+",
        default=[-30, 30],
        help="X Dimension Limits",
    )
    parser.add_argument(
        "--ylims",
        type=float,
        nargs="+",
        default=[-30, 30],
        help="Y Dimension Limits",
    )
    parser.add_argument(
        "--zlims",
        type=float,
        nargs="+",
        default=[-30, 30],
        help="X Dimension Limits",
    )

    return parser.parse_args()


def get_close_pose_idx(vals, rid, pose_index, index_tresh, dist_thresh):
    current_pose = vals.atPose3(gtsam.symbol(rid, pose_index))
    close_pose_indexes = []
    for i in range(pose_index - 1):
        pose = vals.atPose3(gtsam.symbol(rid, i))
        if (
            np.linalg.norm(current_pose.inverse().compose(pose).translation())
            < dist_thresh
            and abs(i - pose_index) > index_tresh
        ):
            close_pose_indexes.append(i)
    if len(close_pose_indexes) > 1:
        return np.random.choice(close_pose_indexes)
    else:
        return None


def get_comm_robot(vals, robots, rid, pose_index, dist_thresh):
    current_pose = vals.atPose3(gtsam.symbol(rid, pose_index))
    shuffled_robots = copy(robots)
    random.shuffle(shuffled_robots)
    for other_rid in shuffled_robots:
        if rid != other_rid:
            pose = vals.atPose3(gtsam.symbol(other_rid, pose_index))
            if (
                np.linalg.norm(current_pose.inverse().compose(pose).translation())
                < dist_thresh
            ):
                return other_rid
    return None


def get_available_comms(vals, robots, pose_index, dist_thresh):
    avaliable = copy(robots)
    comms = []

    for rid in robots:
        if rid in avaliable:
            other_rid = get_comm_robot(vals, avaliable, rid, pose_index, dist_thresh)
            if other_rid:
                comms.append((rid, other_rid))
                avaliable.remove(rid)
                avaliable.remove(other_rid)
    return comms


def add_priors(
    builder,
    robots,
    stamp,
    pose_number,
    prev_gt_poses,
    prev_est_poses,
    args,
):
    for i, rid in enumerate(robots):
        key = gtsam.symbol(rid, pose_number)
        fg = gtsam.NonlinearFactorGraph()
        # Determine the initial Pose
        initial_rot = np.random.choice(ODOM_OPTIONS_GRIDWORLD).rotation()
        initial_position = np.array(
            [
                np.random.uniform(args.xlims[0] / 2, args.xlims[1] / 2),
                np.random.uniform(args.ylims[0] / 2, args.ylims[1] / 2),
                np.random.uniform(args.zlims[0] / 2, args.zlims[1] / 2),
            ]
        )
        if i == 0:
            initial_position = np.zeros(3)
            initial_rot = gtsam.Rot3()
        init_pose = gtsam.Pose3(initial_rot, initial_position)

        noise_sigmas = copy(args.prior_noise_sigmas)
        if i == 0:
            noise_sigmas = copy(args.robot_zero_prior_noise_sigmas)

        # Add as factor
        fg.addPriorPose3(
            key,
            init_pose,
            gtsam.noiseModel.Isotropic.Sigmas(noise_sigmas),
        )
        vals = gtsam.Values()
        vals.insert(key, init_pose)

        builder.addEntry(
            rid,
            stamp,
            fg,
            [jrl.PriorFactorPose3Tag],
            {},
            jrl.TypedValues(vals, {key: jrl.Pose3Tag}),
            jrl.TypedValues(vals, {key: jrl.Pose3Tag}),
        )
        # Update the prev_vals
        prev_gt_poses.insert(key, init_pose)
        prev_est_poses.insert(key, init_pose)
    return builder, prev_gt_poses, prev_est_poses


def add_odom_step(
    rid,
    graphs,
    factor_types,
    new_est_vals,
    new_est_types,
    new_gt_vals,
    new_gt_types,
    pose_number,
    odom_noise_model,
    odom_noise_gen,
    odom_gen,
    prev_gt_poses,
    prev_est_poses,
    init_noise_gen,
    args,
):
    key = gtsam.symbol(rid, pose_number)
    prev_key = gtsam.symbol(rid, pose_number - 1)

    odom = odom_gen(prev_gt_poses.atPose3(prev_key))
    noise = odom_noise_gen()
    measure = odom.compose(noise)

    gt_pose = prev_gt_poses.atPose3(prev_key).compose(odom)
    est_pose = prev_est_poses.atPose3(prev_key).compose(measure)

    graphs[rid].add(gtsam.BetweenFactorPose3(prev_key, key, measure, odom_noise_model))
    factor_types[rid].append(jrl.BetweenFactorPose3Tag)

    new_est_vals[rid].insert(key, est_pose)
    new_est_types[rid][key] = jrl.Pose3Tag

    new_gt_vals[rid].insert(key, gt_pose)
    new_gt_types[rid][key] = jrl.Pose3Tag

    # Update the prev_vals
    prev_gt_poses.insert(key, gt_pose)
    if args.initialization_type == "gt":
        prev_est_poses.insert(key, gt_pose)
    elif args.initialization_type == "noisy_gt":
        prev_est_poses.insert(key, gt_pose.compose(init_noise_gen()))
    elif args.initialization_type == "odom":
        prev_est_poses.insert(key, est_pose)
    else:
        raise Exception("Invalid Initialization_type")


def make_intra_loop_entry(vals, k1, k2, odom_noise_model, gen_noise):
    noise = gen_noise()
    measure = vals.atPose3(k1).inverse().compose(vals.atPose3(k2)).compose(noise)
    return gtsam.BetweenFactorPose3(k1, k2, measure, odom_noise_model)


def make_loop_entry(vals, k1, k2, noise_model, gen_noise, measure_type):
    noise = gen_noise()
    if measure_type == "pose":
        measure = vals.atPose3(k1).inverse().compose(vals.atPose3(k2)).compose(noise)
        return gtsam.BetweenFactorPose3(k1, k2, measure, noise_model)
    elif measure_type == "range":
        raise Exception("range not enabled for 3d")


def add_self_loops(
    rid,
    stamp,
    graphs,
    factor_types,
    loop_noise_model,
    loop_noise_gen,
    gt_poses,
    args,
):
    close_pose_idx = get_close_pose_idx(
        gt_poses,
        rid,
        stamp,
        args.loop_closure_index_threshold,
        args.loop_closure_distance_threshold,
    )
    if close_pose_idx and np.random.rand() < args.loop_closure_probability:
        key = gtsam.symbol(rid, stamp)
        prev_key = gtsam.symbol(rid, close_pose_idx)

        graphs[rid].add(
            make_loop_entry(
                gt_poses, key, prev_key, loop_noise_model, loop_noise_gen, "pose"
            )
        )
        factor_types[rid].append(jrl.BetweenFactorPose3Tag)


def add_comm_loops(
    comms,
    rid,
    stamp,
    graphs,
    factor_types,
    new_est_vals,
    new_est_types,
    new_gt_vals,
    new_gt_types,
    comm_loop_noise_model,
    comm_loop_noise_gen,
    gt_poses,
    est_poses,
    args,
):

    if args.comm_loop_measurement_type == "pose":
        measure_type_tag = jrl.BetweenFactorPose3Tag
    elif args.comm_loop_measurement_type == "range":
        measure_type_tag = jrl.RangeFactorPose3Tag
    elif args.comm_loop_measurement_type == "bearing_range":
        measure_type_tag = jrl.BearingRangeFactorPose3Tag

    for ra, rb in comms:
        ka = gtsam.symbol(ra, stamp)
        kb = gtsam.symbol(rb, stamp)

        # Update for Robot A
        graphs[ra].add(
            make_loop_entry(
                gt_poses,
                ka,
                kb,
                comm_loop_noise_model,
                comm_loop_noise_gen,
                args.comm_loop_measurement_type,
            )
        )
        factor_types[ra].append(measure_type_tag)

        new_est_vals[ra].insert(kb, est_poses.atPose3(kb))
        new_est_types[ra][kb] = jrl.Pose3Tag

        new_gt_vals[ra].insert(kb, gt_poses.atPose3(kb))
        new_gt_types[ra][kb] = jrl.Pose3Tag

        # Update for Robot B
        graphs[rb].add(
            make_loop_entry(
                gt_poses,
                kb,
                ka,
                comm_loop_noise_model,
                comm_loop_noise_gen,
                args.comm_loop_measurement_type,
            )
        )
        factor_types[rb].append(measure_type_tag)

        new_est_vals[rb].insert(ka, est_poses.atPose3(ka))
        new_est_types[rb][ka] = jrl.Pose3Tag

        new_gt_vals[rb].insert(ka, gt_poses.atPose3(ka))
        new_gt_types[rb][ka] = jrl.Pose3Tag


def make_dataset(args, dataset_count):
    # Setup ID's for each robot
    robots = []
    for i in range(args.number_robots):
        robots.append(ascii_letters[i])

    # Setup the Dataset Builder
    builder = jrl.DatasetBuilder(args.name + "_{:04d}".format(dataset_count), robots)

    # Setup the noise Models
    odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas(args.odom_noise_sigmas)

    def odom_noise_gen():
        return gtsam.Pose3.Expmap(
            np.random.multivariate_normal(
                np.zeros((6,)), np.diag(np.array(args.odom_noise_sigmas) ** 2)
            )
        )

    loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(args.loop_noise_sigmas)

    def loop_noise_gen():
        return gtsam.Pose3.Expmap(
            np.random.multivariate_normal(
                np.zeros((6,)), np.diag(np.array(args.loop_noise_sigmas) ** 2)
            )
        )

    def init_noise_gen():
        return gtsam.Pose3.Expmap(
            np.random.multivariate_normal(
                np.zeros((6,)), np.diag(np.array(args.initialization_noise_sigmas) ** 2)
            )
        )

    if args.comm_loop_measurement_type == "pose":
        comm_loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            args.comm_loop_noise_sigmas
        )
    elif args.comm_loop_measurement_type == "range":
        comm_loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            [args.comm_loop_noise_sigmas[0]]
        )
    else:
        raise RuntimeError("Invalid comm_loop_measurement_type provided")

    def comm_loop_noise_gen():
        if args.comm_loop_measurement_type == "pose":
            return gtsam.Pose3.Expmap(
                np.random.multivariate_normal(
                    np.zeros((6,)), np.diag(np.array(args.comm_loop_noise_sigmas) ** 2)
                )
            )
        elif args.comm_loop_measurement_type == "range":
            return np.random.normal(0, args.comm_loop_noise_sigmas[0])

    # Setup the Odometry Model:
    def gen_odom(pose):
        i = np.random.choice(np.arange(len(args.odom_probs)), p=args.odom_probs)
        if pose.x() < args.xlims[0]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.Identity(), np.array([args.xlims[0], pose.y(), pose.z()])
            )
            return pose.inverse().compose(end_pose)
        elif pose.x() > args.xlims[1]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.RzRyRx(np.pi, 0, 0),
                np.array([args.xlims[1], pose.y(), pose.z()]),
            )
            return pose.inverse().compose(end_pose)
        elif pose.y() < args.ylims[0]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.RzRyRx(np.pi / 2, 0, 0),
                np.array([pose.x(), args.ylims[0], pose.z()]),
            )
            return pose.inverse().compose(end_pose)
        elif pose.y() > args.ylims[1]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.RzRyRx(-np.pi / 2, 0, 0),
                np.array([pose.x(), args.ylims[1], pose.z()]),
            )
            return pose.inverse().compose(end_pose)
        elif pose.z() < args.zlims[0]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.RzRyRx(0, -np.pi / 2, 0),
                np.array([pose.x(), pose.y(), args.zlims[0]]),
            )
            return pose.inverse().compose(end_pose)
        elif pose.z() > args.zlims[1]:
            end_pose = gtsam.Pose3(
                gtsam.Rot3.RzRyRx(0, np.pi / 2, 0),
                np.array([pose.x(), pose.y(), args.zlims[1]]),
            )
            return pose.inverse().compose(end_pose)
        else:
            return ODOM_OPTIONS_GRIDWORLD[i]

    stamp = 0
    gtvals = gtsam.Values()
    initvals = gtsam.Values()

    builder, gtvals, initvals = add_priors(
        builder, robots, stamp, 0, gtvals, initvals, args
    )
    for pose_num in range(1, args.number_poses):
        stamp += 1

        graphs = {}
        factor_types = {}
        new_est_vals = {}
        new_est_types = {}
        new_gt_vals = {}
        new_gt_types = {}
        for rid in robots:
            graphs[rid] = gtsam.NonlinearFactorGraph()
            factor_types[rid] = []
            new_est_vals[rid] = gtsam.Values()
            new_est_types[rid] = {}
            new_gt_vals[rid] = gtsam.Values()
            new_gt_types[rid] = {}

        # First generate odometry
        for rid in robots:
            add_odom_step(
                rid,
                graphs,
                factor_types,
                new_est_vals,
                new_est_types,
                new_gt_vals,
                new_gt_types,
                stamp,
                odom_noise_model,
                odom_noise_gen,
                gen_odom,
                gtvals,
                initvals,
                init_noise_gen,
                args,
            )

        # Second generate loop-closures
        for rid in robots:
            add_self_loops(
                rid,
                stamp,
                graphs,
                factor_types,
                loop_noise_model,
                loop_noise_gen,
                gtvals,
                args,
            )

        # finally generate inter-robot measurements
        if pose_num % args.comm_freq == 0:
            comms = get_available_comms(gtvals, robots, stamp, args.comm_range)
            add_comm_loops(
                comms,
                rid,
                stamp,
                graphs,
                factor_types,
                new_est_vals,
                new_est_types,
                new_gt_vals,
                new_gt_types,
                comm_loop_noise_model,
                comm_loop_noise_gen,
                gtvals,
                initvals,
                args,
            )

        for rid in robots:
            builder.addEntry(
                rid,
                stamp,
                graphs[rid],
                factor_types[rid],
                {},
                jrl.TypedValues(new_est_vals[rid], new_est_types[rid]),
                jrl.TypedValues(new_gt_vals[rid], new_gt_types[rid]),
            )

    dataset = builder.build()
    writer = jrl.Writer()
    writer.writeDataset(
        dataset,
        os.path.join(args.output_dir, args.name + "_{:04d}.jrl".format(dataset_count)),
        False,
    )


def main():
    args = handle_args()
    for i in range(args.repeats):
        print(i)
        make_dataset(args, i)


if __name__ == "__main__":
    main()