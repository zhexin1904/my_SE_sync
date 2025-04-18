#!/usr/bin/env python
import argparse
import os
import sys
from datetime import date

import random
import gtsam
import jrl
import numpy as np
from gtsam.symbol_shorthand import X
from scipy.stats import chi2
from copy import copy
from string import ascii_letters


# GLOBALS
ODOM_OPTIONS_GRIDWORLD = [
    gtsam.Pose2(1, 0, 0),  # Move forward
    gtsam.Pose2(0, 0, np.pi / 2.0),  # Turn left
    gtsam.Pose2(0, 0, -np.pi / 2.0),  # Turn right
]

# GLOBALS
ODOM_OPTIONS_CONTINUOUS = [
    gtsam.Pose2(0.5, 0, 0),  # Move forward
    gtsam.Pose2(0.5, 0, 0.4),  # Turn left
    gtsam.Pose2(0.5, 0, -0.4),  # Turn right
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
        default=[0.8, 0.15, 0.05],
        help="Odom action probabilities [forward, left, right]",
    )
    parser.add_argument(
        "--odom_type",
        type=str,
        default="gridworld",
        help="Odometry Type [gridworld, contworld] (grid or continuous space",
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
        default=[10, 10, 1],
        help="Sigmas for diagonal noise model of prior measurements",
    )

    parser.add_argument(
        "--robot_zero_prior_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.01, 0.01, 1],
        help="Sigmas for diagonal noise model of robot zero's prior measurements",
    )

    parser.add_argument(
        "--odom_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.05, 1],
        help="Sigmas for diagonal noise model of odometry measurements",
    )

    parser.add_argument(
        "--loop_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.05, 1],
        help="Sigmas for diagonal noise model of intra-robot loop closure measurements",
    )

    parser.add_argument(
        "--comm_loop_measurement_type",
        type=str,
        default="pose",
        help="The measurement type for inter-robot measurements [pose, range, bearing_range]",
    )

    parser.add_argument(
        "--comm_loop_noise_sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.05, 1],
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
        default=[1.0, 1.0, 20],
        help="Sigmas to use for 'noisy-gt' initialization.",
    )

    parser.add_argument(
        "--xlims",
        type=float,
        nargs="+",
        default=[-100, 100],
        help="X Dimension Limits",
    )
    parser.add_argument(
        "--ylims",
        type=float,
        nargs="+",
        default=[-100, 100],
        help="X Dimension Limits",
    )

    return parser.parse_args()


def get_close_pose_idx(vals, rid, pose_index, index_tresh, dist_thresh):
    current_pose = vals.atPose2(gtsam.symbol(rid, pose_index))
    cx, cy = current_pose.x(), current_pose.y()
    for i in range(pose_index - 1):
        pose = vals.atPose2(gtsam.symbol(rid, i))
        x, y = pose.x(), pose.y()
        if (
            np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < dist_thresh
            and abs(i - pose_index) > index_tresh
        ):
            return i
    return None


def get_comm_robot(vals, robots, rid, pose_index, dist_thresh):
    current_pose = vals.atPose2(gtsam.symbol(rid, pose_index))
    cx, cy = current_pose.x(), current_pose.y()
    shuffled_robots = copy(robots)
    random.shuffle(shuffled_robots)
    for other_rid in shuffled_robots:
        if rid != other_rid:
            pose = vals.atPose2(gtsam.symbol(other_rid, pose_index))
            x, y = pose.x(), pose.y()
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < dist_thresh:
                return other_rid
    return None


def get_available_comms(vals, robots, pose_index, dist_thresh):
    avaliable = copy(robots)
    comms = []

    search = copy(robots)
    random.shuffle(search)
    for rid in search:
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
        if args.odom_type == "contworld":
            start_y = [-10, -5, 0, 5, 10]
            init_pose = gtsam.Pose2(0, start_y[i] / 5, 0)  # np.random.uniform(0, 0.1))
        else:
            init_pose = gtsam.Pose2(
                np.random.uniform(args.xlims[0] * 0.8, args.xlims[1] * 0.8),
                np.random.uniform(args.ylims[0] * 0.8, args.ylims[1] * 0.8),
                np.random.choice([0, np.pi / 2, np.pi, -np.pi / 2]),
            )

        noise_sigmas = copy(args.prior_noise_sigmas)
        if i == 0:
            noise_sigmas = copy(args.robot_zero_prior_noise_sigmas)
        # Convert theta element to radians
        noise_sigmas[2] = np.deg2rad(noise_sigmas[2])

        fg.addPriorPose2(
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
            [jrl.PriorFactorPose2Tag],
            {},
            jrl.TypedValues(vals, {key: jrl.Pose2Tag}),
            jrl.TypedValues(vals, {key: jrl.Pose2Tag}),
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

    odom = odom_gen(prev_gt_poses.atPose2(prev_key))
    noise = odom_noise_gen()
    measure = odom.compose(noise)

    gt_pose = prev_gt_poses.atPose2(prev_key).compose(odom)
    est_pose = prev_est_poses.atPose2(prev_key).compose(measure)

    graphs[rid].add(gtsam.BetweenFactorPose2(prev_key, key, measure, odom_noise_model))
    factor_types[rid].append(jrl.BetweenFactorPose2Tag)

    new_est_vals[rid].insert(key, est_pose)
    new_est_types[rid][key] = jrl.Pose2Tag

    new_gt_vals[rid].insert(key, gt_pose)
    new_gt_types[rid][key] = jrl.Pose2Tag

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
    measure = vals.atPose2(k1).inverse().compose(vals.atPose2(k2)).compose(noise)
    return gtsam.BetweenFactorPose2(k1, k2, measure, odom_noise_model)


def make_loop_entry(vals, k1, k2, noise_model, gen_noise, measure_type):
    noise = gen_noise()
    if measure_type == "pose":
        measure = vals.atPose2(k1).inverse().compose(vals.atPose2(k2)).compose(noise)
        return gtsam.BetweenFactorPose2(k1, k2, measure, noise_model)
    elif measure_type == "range":
        rel_pose = vals.atPose2(k1).inverse().compose(vals.atPose2(k2))
        r = r = np.linalg.norm(rel_pose.translation()) + noise
        return gtsam.RangeFactorPose2(k1, k2, r, noise_model)
    elif measure_type == "bearing_range":
        rel_pose = vals.atPose2(k1).inverse().compose(vals.atPose2(k2))
        b = gtsam.Rot2(np.arctan2(rel_pose.y(), rel_pose.x()) + noise[0])
        r = np.linalg.norm(rel_pose.translation()) + noise[1]
        return gtsam.BearingRangeFactorPose2(k1, k2, b, r, noise_model)


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
        factor_types[rid].append(jrl.BetweenFactorPose2Tag)


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
        measure_type_tag = jrl.BetweenFactorPose2Tag
    elif args.comm_loop_measurement_type == "range":
        measure_type_tag = jrl.RangeFactorPose2Tag
    elif args.comm_loop_measurement_type == "bearing_range":
        measure_type_tag = jrl.BearingRangeFactorPose2Tag

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

        new_est_vals[ra].insert(kb, est_poses.atPose2(kb))
        new_est_types[ra][kb] = jrl.Pose2Tag

        new_gt_vals[ra].insert(kb, gt_poses.atPose2(kb))
        new_gt_types[ra][kb] = jrl.Pose2Tag

        # Note Comment Out for Bayes Tree Example figure
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

        new_est_vals[rb].insert(ka, est_poses.atPose2(ka))
        new_est_types[rb][ka] = jrl.Pose2Tag

        new_gt_vals[rb].insert(ka, gt_poses.atPose2(ka))
        new_gt_types[rb][ka] = jrl.Pose2Tag


def make_dataset(args, dataset_count):
    # Setup ID's for each robot
    robots = []
    for i in range(args.number_robots):
        robots.append(ascii_letters[i])

    # Setup the Dataset Builder
    builder = jrl.DatasetBuilder(args.name + "_{:04d}".format(dataset_count), robots)

    # Setup the noise Models
    odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        [
            args.odom_noise_sigmas[0],
            args.odom_noise_sigmas[1],
            np.deg2rad(args.odom_noise_sigmas[2]),
        ]
    )

    def odom_noise_gen():
        return gtsam.Pose2(
            np.random.normal(0, args.odom_noise_sigmas[0]),
            np.random.normal(0, args.odom_noise_sigmas[1]),
            np.random.normal(0, np.deg2rad(args.odom_noise_sigmas[2])),
        )

    loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        [
            args.loop_noise_sigmas[0],
            args.loop_noise_sigmas[1],
            np.deg2rad(args.loop_noise_sigmas[2]),
        ]
    )

    def loop_noise_gen():
        return gtsam.Pose2(
            np.random.normal(0, args.loop_noise_sigmas[0]),
            np.random.normal(0, args.loop_noise_sigmas[1]),
            np.random.normal(0, np.deg2rad(args.loop_noise_sigmas[2])),
        )

    def init_noise_gen():
        return gtsam.Pose2(
            np.random.normal(0, args.initialization_noise_sigmas[0]),
            np.random.normal(0, args.initialization_noise_sigmas[1]),
            np.random.normal(0, np.deg2rad(args.initialization_noise_sigmas[2])),
        )

    if args.comm_loop_measurement_type == "pose":
        comm_loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            [
                args.odom_noise_sigmas[0],
                args.odom_noise_sigmas[1],
                np.deg2rad(args.odom_noise_sigmas[2]),
            ]
        )
    elif args.comm_loop_measurement_type == "range":
        comm_loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            [args.comm_loop_noise_sigmas[0]]
        )
    elif args.comm_loop_measurement_type == "bearing_range":
        comm_loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            [np.deg2rad(args.comm_loop_noise_sigmas[0]), args.comm_loop_noise_sigmas[1]]
        )
    else:
        raise RuntimeError("Invalid comm_loop_measurement_type provided")

    def comm_loop_noise_gen():
        if args.comm_loop_measurement_type == "pose":
            return gtsam.Pose2(
                np.random.normal(0, args.comm_loop_noise_sigmas[0]),
                np.random.normal(0, args.comm_loop_noise_sigmas[1]),
                np.random.normal(0, np.deg2rad(args.comm_loop_noise_sigmas[2])),
            )
        elif args.comm_loop_measurement_type == "range":
            return np.random.normal(0, args.comm_loop_noise_sigmas[0])
        elif args.comm_loop_measurement_type == "bearing_range":
            return np.array(
                [
                    np.random.normal(0, np.deg2rad(args.comm_loop_noise_sigmas[0])),
                    np.random.normal(0, args.comm_loop_noise_sigmas[1]),
                ]
            )

    # Setup the Odometry Model:
    def gen_odom(pose):
        i = np.random.choice([0, 1, 2], p=args.odom_probs)
        if pose.x() < args.xlims[0]:
            return pose.inverse().compose(gtsam.Pose2(args.xlims[0], pose.y(), 0))
        elif pose.x() > args.xlims[1]:
            return pose.inverse().compose(gtsam.Pose2(args.xlims[1], pose.y(), np.pi))
        elif pose.y() < args.ylims[0]:
            return pose.inverse().compose(
                gtsam.Pose2(pose.x(), args.ylims[0], np.pi / 2.0)
            )
        elif pose.y() > args.ylims[1]:
            return pose.inverse().compose(
                gtsam.Pose2(pose.x(), args.ylims[1], -np.pi / 2.0)
            )

        if args.odom_type == "gridworld":
            return ODOM_OPTIONS_GRIDWORLD[i]
        elif args.odom_type == "contworld":
            return ODOM_OPTIONS_CONTINUOUS[i]
        else:
            raise RuntimeError("Invalid Odometry Type: {}".format(args.odom_type))

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