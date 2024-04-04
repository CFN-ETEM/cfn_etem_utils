import glob, os, cv2, math, socket, copy, bz2
import argparse
import json
import re
from itertools import chain
from scipy.spatial import distance_matrix
import numpy as np

from cfntem.particle_tracking.tracking import ParticleTracker
import ipyparallel as ipp


def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg


def directory_to_create(parser, arg):
    if os.path.exists(arg):
        if not os.path.isdir(arg):
            parser.error('The directory {} does not exist!'.format(arg))
    else:
        os.makedirs(arg)
    return arg


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def init_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains cropped binary image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-c", "--checkpoints", help="The folder to holds the statistics file",
                        required=True,
                        type=lambda x: directory_to_create(parser, x))
    parser.add_argument("--raw_in", help="The folder contain the cropped raw image before track",
                        required=False,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("--raw_out", help="The folder to put cropped raw image",
                        required=False,
                        type=lambda x: directory_to_create(parser, x))
    parser.add_argument("-m", "--max_history", help="maximum frames in history",
                        default=100,
                        type=int)
    parser.add_argument("-a", "--area_min", help="minimum area for the particle to be considered",
                        default=40,
                        type=int)
    parser.add_argument("--margin", help="the center of particle should be away from the border",
                        default=15,
                        type=int)
    parser.add_argument("-g", "--min_gap", help="Minimum to the current particle to keep a particle in history",
                        default=3,
                        type=int)
    parser.add_argument("--suspicious_move", help="The threshold to check whether the move of the particle is suspicious",
                        default=20,
                        type=int)
    parser.add_argument("--suspicious_rad_change",
                        help="The threshold to check if there is sudden change in particle radius",
                        default=5,
                        type=int)
    parser.add_argument("--suspicious_confirm_frames",
                        help="Number of consecutive frames to confirm the suspicious change is real",
                        default=5,
                        type=int)
    parser.add_argument("-p", "--parallel", help="Run the job in parallel mode",
                        action='store_true')
    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)
    checkpoint_dir = os.path.abspath(args.checkpoints)
    raw_dir_in = args.raw_in
    raw_dir_out = args.raw_out
    max_history = args.max_history
    min_area = args.area_min
    margin = args.margin
    parallel_run = args.parallel
    min_gap = args.min_gap
    suspicious_move = args.suspicious_move
    suspicious_rad_change = args.suspicious_rad_change
    suspicious_confirm_frames = args.suspicious_confirm_frames

    in_file_list = list(chain(*[glob.glob("{}/Capture{}/**/*.png".format(source_dir, "[0-9]" * n), recursive=True)
                                for n in range(1, 4)]))
    if len(in_file_list) == 0:
        in_file_list = glob.glob("{}/**/*.png".format(source_dir), recursive=True)
    if len(in_file_list) == 0:
        print("Error: no files found")
        exit(1)
    in_file_list = sorted(in_file_list, key=lambda x: (
        int(re.search(r'Capture(\d+).*', x).group(1)),
        x))

    return parallel_run, source_dir, checkpoint_dir, raw_dir_in, raw_dir_out, in_file_list, max_history, min_area, \
           margin, suspicious_move, suspicious_rad_change, suspicious_confirm_frames, min_gap


def sub_job(source_dir, checkpoint_dir, raw_dir_in, raw_dir_out, in_file_list, max_history, min_area, margin,
            suspicious_move, suspicious_rad_change, suspicious_confirm_frames, min_gap, starting_frame):
    frame2fnames = dict()
    tracker = ParticleTracker(max_history, min_area, margin, suspicious_move, suspicious_rad_change,
                              suspicious_confirm_frames, min_gap)
    last_chkpt_fn = None
    for i, fn in enumerate(in_file_list):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        frame2fnames[starting_frame + tracker.current_frame_number] = fn
        tracker.track(img)
        if i % 10 == 0:
            chkpt_fn = tracker.save_checkpoint(os.path.abspath(os.path.join(checkpoint_dir,
                    "checkpoints.pid{}.host{}".format(os.getpid(), socket.gethostname()))))
            if last_chkpt_fn is not None and os.path.exists(last_chkpt_fn):
                os.remove(last_chkpt_fn)
            last_chkpt_fn = chkpt_fn
        if raw_dir_in is not None and raw_dir_out is not None:
            raw_fn_in = os.path.abspath(fn).replace(source_dir, raw_dir_in)
            if os.path.exists(raw_fn_in):
                cropped_raw_fn = os.path.abspath(fn).replace(source_dir, raw_dir_out)
                img_raw_in = cv2.imread(raw_fn_in, -1)
                img_raw_marked = tracker.mark_contours(img_raw_in)
                raw_out_dir = os.path.dirname(cropped_raw_fn)
                if not os.path.exists(raw_out_dir):
                    os.makedirs(raw_out_dir)
                cv2.imwrite(cropped_raw_fn, img_raw_marked, [cv2.IMWRITE_PNG_COMPRESSION, 8])
    tracker.save_checkpoint(os.path.abspath(os.path.join(checkpoint_dir, "final.checkpoints.pid{}.host{}"
                                                         .format(os.getpid(), socket.gethostname()))))
    with open(os.path.abspath(os.path.join(
            checkpoint_dir, "frame2fnames.pid{}.host{}.json"
                    .format(os.getpid(), socket.gethostname()))), 'w') as f:
        json.dump(frame2fnames, f, indent=4, sort_keys=True)
    particles = tracker.particles if tracker.particles is not None else []
    return frame2fnames, particles, tracker.initial_number_particles


def get_parallel_map_func():
    c = ipp.Client(connection_info="ipypar/security/ipcontroller-client.json")
    print(c.ids)
    with c[:].sync_imports():
        import cv2, os, json, bz2
        import socket
        from cfntem.particle_tracking.tracking import ParticleTracker

    return c[:].map_sync, len(c.ids)


def concatenate_trajectory(particles_old, particles_new, dup_frame, max_history, min_area):
    min_rad = math.sqrt(min_area / math.pi)
    last_frame_number = max([traj[-1][0] for traj in particles_old])
    first_frame_new_born_particles = [(traj[0], i) for i, traj in enumerate(particles_new) if
                                      traj[0][0] == 0 and len(traj) > 2]
    buffer_new_born_particles = [(traj[0], i) for i, traj in enumerate(particles_new)
                                 if 0 < traj[0][0] < max_history - 1 and len(traj) > 2]
    last_frame_old_particles = dup_frame
    buffer_old_particles = [(traj[-1], i) for i, traj in enumerate(particles_old)
                            if last_frame_number - max_history + 1 < traj[-1][0]]

    first_frame_new_coords = [p[0][4:6] for p in first_frame_new_born_particles]
    last_frame_old_coords = [p[0][4:6] for p in last_frame_old_particles]

    single_to_group_id = dict()

    if len(last_frame_old_coords) > 0 and len(first_frame_new_coords) > 0:
        dm1 = distance_matrix(last_frame_old_coords, first_frame_new_coords)
        for prev_id_l1, new_id_l1 in zip(*np.where(dm1 < min_rad)):
            single_id = first_frame_new_born_particles[new_id_l1][1]
            group_id = last_frame_old_particles[prev_id_l1][1]
            if group_id not in single_to_group_id.values():
                single_to_group_id[single_id] = group_id

    buffer_new_coords = [p[0][4:6] for p in buffer_new_born_particles]
    buffer_old_coords = [p[0][4:6] for p in buffer_old_particles]
    if len(buffer_old_coords) >= 1 and len(buffer_new_coords) >= 1:
        dm2 = distance_matrix(buffer_old_coords, buffer_new_coords)
        for prev_id_l1, new_id_l1 in zip(*np.where(dm2 < 100)):
            single_id = buffer_new_born_particles[new_id_l1][1]
            group_id = buffer_old_particles[prev_id_l1][1]

            single_frame_number = buffer_new_born_particles[new_id_l1][0][0]
            group_frame_number = buffer_old_particles[prev_id_l1][0][0]
            new_radius = math.sqrt(particles_new[single_id][0][3] / math.pi)
            old_radius = math.sqrt(particles_old[group_id][-1][3] / math.pi)
            if dm2[prev_id_l1, new_id_l1] < new_radius + old_radius:
                if last_frame_number - group_frame_number + single_frame_number + 1 < max_history:
                    if group_id not in single_to_group_id.values() and single_id not in single_to_group_id:
                        single_to_group_id[single_id] = group_id

    all_new_born_ids = list(range(len(particles_new)))
    unmatched_ids = set(all_new_born_ids) - set(single_to_group_id.keys())
    cur_id = len(particles_old)
    for i in sorted(unmatched_ids):
        if len(particles_new[i]) > 2:
            single_to_group_id[i] = cur_id
            cur_id += 1
    return single_to_group_id, cur_id - len(particles_old)


def main():
    parallel_run, source_dir, checkpoint_dir, raw_dir_in, raw_dir_out, full_in_file_list, max_history, min_area, \
    margin, suspicious_move, suspicious_rad_change, suspicious_confirm_frames, min_gap = init_params()
    if parallel_run:
        map, nprocesses = get_parallel_map_func()
    else:
        map, nprocesses = map, 1

    print("running with {} processes".format(nprocesses))

    nfiles_per_process = math.ceil(len(full_in_file_list) / nprocesses)
    ifl_chunks = [full_in_file_list[i * nfiles_per_process: (i + 1) * nfiles_per_process]
                  for i in range(nprocesses)]
    orig_ifl_chunck = copy.deepcopy(ifl_chunks)
    for i in range(nprocesses - 1):
        if len(ifl_chunks[i + 1]) > 0:
            ifl_chunks[i].append(ifl_chunks[i + 1][0])

    starting_frames = [i * nfiles_per_process for i in range(nprocesses)]

    track_data = map(sub_job,
                     [source_dir] * nprocesses,
                     [checkpoint_dir] * nprocesses,
                     [raw_dir_in] * nprocesses,
                     [raw_dir_out] * nprocesses,
                     ifl_chunks,
                     [max_history] * nprocesses,
                     [min_area] * nprocesses,
                     [margin] * nprocesses,
                     [suspicious_move] * nprocesses,
                     [suspicious_rad_change] * nprocesses,
                     [suspicious_confirm_frames] * nprocesses,
                     [min_gap] * nprocesses,
                     starting_frames)

    track_data = list(track_data)

    if nprocesses > 1:
        frame2fnames, concatenated_particles, initial_number_particles = dict(), list(), track_data[0][-1]
        dup_frame = None
        for i, (chunk_f2f, chunk_particles, _) in enumerate(track_data):
            frame2fnames.update(chunk_f2f)
            if i == 0:
                single_to_group_id = dict()
                for j, p in enumerate(chunk_particles):
                    if len(p) > 2:
                        single_to_group_id[j] = len(concatenated_particles)
                        concatenated_particles.append(p[:-1])
                dup_frame = [(p[-1], single_to_group_id[j]) for j, p in enumerate(chunk_particles) if len(p) > 2]
            else:
                single_to_group_id, n_new = concatenate_trajectory(concatenated_particles, chunk_particles, dup_frame, max_history, min_area)
                for _ in range(n_new):
                    concatenated_particles.append(list())
                for j, traj in enumerate(chunk_particles):
                    if len(traj) > 2:
                        new_id = single_to_group_id[j]
                        if i == nprocesses - 1 or len(ifl_chunks[i+1]) == 0:
                            truc_traj = traj
                        else:
                            truc_traj = traj[:-1]
                        for frame_number, circle_x, circle_y, area, centroid_x, centroid_y, gyradius in truc_traj:
                            concatenated_particles[new_id].append([frame_number + starting_frames[i],
                                                                   circle_x, circle_y, area, centroid_x,
                                                                   centroid_y, gyradius])
                dup_frame = [([p[-1][0] + starting_frames[i]] + p[-1][1:], single_to_group_id[k]) for k, p in
                             enumerate(chunk_particles)
                             if len(p) > 2]
    else:
        frame2fnames, concatenated_particles, initial_number_particles = track_data[0]


    print("There are {} particles in the initial frame".format(initial_number_particles))
    print("{} particles in the whole process".format(len(concatenated_particles)))
    print("Total {} new particle emerged in the tracking process".format(len(concatenated_particles) - initial_number_particles))

    with bz2.open("particle_stats.json.bz2", 'wt') as f:
        json.dump(concatenated_particles, f, indent=4)

    with open("frame2fnames.json", 'w') as f:
        json.dump(frame2fnames, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
