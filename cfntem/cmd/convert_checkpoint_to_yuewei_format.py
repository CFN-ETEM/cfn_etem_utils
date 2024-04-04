import json, argparse
import os

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg

def fill_frame(i, j, frame, frame_coordi, centroids, particle_areas, gyradii):
    _, circle_x, circle_y, area, centroid_x, centroid_y, gyradius = frame
    frame_coordi[i][j] = (circle_x, circle_y)
    particle_areas[i][j] = area
    centroids[i][j] = (centroid_x, centroid_y)
    gyradii[i][j] = gyradius


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="The checkpoint file",
                        required=True,
                        type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    chkpt_file = os.path.abspath(args.checkpoint)

    with open(chkpt_file, 'r') as f:
        chkpt = json.load(f)

    num_tracks = len(chkpt)
    num_frames = max([p[-1][0] for p in chkpt]) + 1

    frame_coordi = [([([0] * 2) for p in range(num_frames)]) for q in range(num_tracks)]
    centroids = [([([0] * 2) for p in range(num_frames)]) for q in range(num_tracks)]
    particle_areas = [[0 for p in range(num_frames)] for q in range(num_tracks)]
    gyradii = [[0 for p in range(num_frames)] for q in range(num_tracks)]

    for i, frames in enumerate(chkpt):
        last_frame = None
        for frame in frames:
            j = frame[0]
            fill_frame(i, j, frame, frame_coordi, centroids, particle_areas, gyradii)
            if last_frame is not None:
                for cj in range(last_j + 1, j):
                    fill_frame(i, cj, last_frame, frame_coordi, centroids, particle_areas, gyradii)
            last_j = j
            last_frame = frame

    number_particles = [0] * num_frames

    for i in range(num_tracks):
        for j in range(num_frames):
            if particle_areas[i][j] > 1:
                number_particles[j] += 1

    with open('coordinates.txt', 'w') as txtfile:
        # (circle center x , circle center y, area, centroid x, cnetroid y, gyradius)
        for i in range(num_tracks):
            for j in range(num_frames):
                txtfile.write('(')
                txtfile.write(str(frame_coordi[i][j][0]))
                txtfile.write(',')
                txtfile.write(str(frame_coordi[i][j][1]))
                txtfile.write(',')
                txtfile.write(str(particle_areas[i][j]))
                txtfile.write(',')
                txtfile.write(str(centroids[i][j][0]))
                txtfile.write(',')
                txtfile.write(str(centroids[i][j][1]))
                txtfile.write(',')
                txtfile.write(str(gyradii[i][j]))
                txtfile.write(') ')
            txtfile.write('\n')

    with open('number_of_particles.txt', 'w') as txtfile:
        txtfile.write(', '.join([str(n) for n in number_particles]) + '\n')
