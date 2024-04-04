import cv2, math
import numpy as np
import copy, json, os, bz2
from scipy.spatial import distance_matrix
from collections import defaultdict


class ParticleTracker(object):
    _INDEX_SHIFT = 10
    MAX_NUM_PARTICLES = 10000

    def __init__(self, max_history_frame_num=10, min_area=40, margin=15, suspicious_move=20, suspicious_rad_change=5,
                 suspicious_confirm_frames=5, min_gap=3):
        self.max_history_frame_num = max_history_frame_num
        self.min_area = min_area
        self.leaving_margin = margin
        self.enter_margin = self.leaving_margin * 2
        self.last_img = None
        self.particles = None
        self.last_frame_particle_ids = None
        self.last_frame_contours = None
        self.last_frame_properties = None
        self.suspicious_contours = None
        self.suspicious_particle_max_missing_frames = 1
        self.min_gap = min_gap
        self.current_frame_number = 0
        self.initial_number_particles = 0
        self.suspicious_move = suspicious_move
        self.suspicious_rad_change = suspicious_rad_change
        self.suspicious_confirm_frames = suspicious_confirm_frames


    @staticmethod
    def _compute_particle_properties(cnt, img, frame_number):
        (circle_x, circle_y), _ = cv2.minEnclosingCircle(cnt)
        circle_x, circle_y = round(circle_x, 3), round(circle_y, 3)
        area = cv2.contourArea(cnt)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        pixelpoints = cv2.findNonZero(mask)
        centroid_x, centroid_y = np.mean(pixelpoints[:, :, 0]), np.mean(pixelpoints[:, :, 1])
        cb = np.array([centroid_x, centroid_y])
        centroid_x, centroid_y = round(centroid_x, 3), round(centroid_y, 3)
        rel_pos = (pixelpoints - cb).reshape(len(pixelpoints), 2)
        gyradius = (np.linalg.norm(rel_pos, axis=1, ord=2) ** 2).mean()
        gyradius = round(gyradius, 3)
        # (frame_number, circle center x , circle center y, area, centroid x, cnetroid y, gyradius)
        return [frame_number, circle_x, circle_y, area, centroid_x, centroid_y, gyradius]

    def _find_image_contours_and_properties(self, img):
        real_contours = []
        particle_properties = []
        candidate_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in candidate_contours:
            prop = self._compute_particle_properties(cnt, img, self.current_frame_number)
            assert len(prop) == 7
            area = prop[3]
            x, y = prop[1], prop[2]
            if area > self.min_area and self._center_inside_image((x, y), self.leaving_margin, img):
                real_contours.append(cnt)
                particle_properties.append(prop)

        # removed nested particle defects
        coords = np.array([prop[1:3] for prop in particle_properties])
        area_list = np.array([prop[3] for prop in particle_properties])
        rad_list = np.sqrt(area_list / math.pi)
        nested_ids = []
        for i_cnt, prop in enumerate(particle_properties):
                dm = distance_matrix([prop[1:3]], coords)[0, :]
                dm[i_cnt] = 99999999.9
                enclosing_particle_ids = np.where(dm < rad_list)[0]
                if len(enclosing_particle_ids) > 0:
                    if (rad_list[enclosing_particle_ids] > rad_list[i_cnt] + 1.0E-6).any():
                        # inside another particle
                        nested_ids.append(i_cnt)
        if len(nested_ids) > 0:
            print("internal ids {} with radius {} are nested in other particles".format(
                        ", ".join([str(int(i)) for i in nested_ids]),
                        ", ".join(["{:.1f}".format(float(x)) for x in rad_list[nested_ids]])))
            real_contours = [x for i, x in enumerate(real_contours) if i not in nested_ids]
            particle_properties = [x for i, x in enumerate(particle_properties) if i not in nested_ids]
        return real_contours, particle_properties

    def _center_inside_image(self, center, margin, img):
        x, y = center
        top, bottom, left, right = margin, img.shape[1] - margin, margin, img.shape[0] - margin
        return top < y < bottom and left < x < right

    def rebase(self, img):
        self.current_frame_number = 0
        self.last_img = img
        contours, partilce_properties = self._find_image_contours_and_properties(img)
        particle_ids = []
        self.particles = []
        confirmed_contours = []
        for cnt, pp in zip(contours, partilce_properties):
            pp = self._compute_particle_properties(cnt, img, self.current_frame_number)
            center = (pp[1], pp[2])
            if self._center_inside_image(center, self.enter_margin, img):
                self.particles.append([pp])
                particle_ids.append(len(particle_ids))
                confirmed_contours.append(cnt)
        self.last_frame_particle_ids = copy.deepcopy(particle_ids)
        self.last_frame_contours = copy.deepcopy(confirmed_contours)
        self.last_frame_properties = copy.deepcopy([traj[-1] for traj in self.particles])
        self.suspicious_contours = defaultdict(list)
        self.initial_number_particles = len(particle_ids)
        print("Message: {} particles detected in the initial frame".format(self.initial_number_particles))

    @staticmethod
    def _write_index_to_contours(img_shape, contours, particle_ids):
        mask = np.zeros(img_shape, np.uint16)
        assert len(contours) == len(particle_ids)

        for i, cnt in zip(particle_ids, contours):
            color = i + ParticleTracker._INDEX_SHIFT
            cv2.drawContours(mask, [cnt], 0, color=color, thickness=-1)
        return mask

    def _fill_particle_ids_from_indexed_img(self, particle_ids, cur_img, cur_contours, cur_contour_ids, indexed_img):
        assert cur_img.dtype == np.uint8
        assert len(particle_ids) >= len(cur_contours)
        assert len(cur_contours) == len(cur_contour_ids)
        cur_img = cur_img.astype(np.uint16)
        cur_img[cur_img > 125] = np.iinfo(np.uint16).max
        index_proj_img = cv2.bitwise_and(cur_img, indexed_img)
        new_countours = []
        new_cnt_cur_ids = []
        multiple_parent_ids = dict()
        multiple_child_ids = dict()
        for i, cnt in zip(cur_contour_ids, cur_contours):
            if particle_ids[i] is not None:
                continue
            mask = np.zeros(cur_img.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0, np.iinfo(np.uint8).max, -1)
            pixelpoints = np.transpose(np.nonzero(mask))
            cur_particle_indices_pixelpoints = [index_proj_img[x, y] for x, y in pixelpoints]
            cur_particle_indices = set(cur_particle_indices_pixelpoints)
            if 0 in cur_particle_indices:
                cur_particle_indices.remove(0)
            if len(cur_particle_indices) == 1:
                cur_id = int(list(cur_particle_indices)[0])
                desired_p_id = cur_id - ParticleTracker._INDEX_SHIFT
                if desired_p_id in particle_ids:
                    multiple_child_ids[i] = desired_p_id
                else:
                    particle_ids[i] = desired_p_id
            elif len(cur_particle_indices) == 0:
                new_countours.append(cnt)
                new_cnt_cur_ids.append(i)
            else:
                potential_ids, id_counts = np.unique(cur_particle_indices_pixelpoints, return_counts=True)
                best_id = int(potential_ids[np.argmax(id_counts)] - ParticleTracker._INDEX_SHIFT)
                while best_id in particle_ids and (id_counts > 0).any():
                    id_counts[potential_ids == best_id + ParticleTracker._INDEX_SHIFT] = -1
                    best_id = int(potential_ids[np.argmax(id_counts)] - ParticleTracker._INDEX_SHIFT)
                else:
                    new_countours.append(cnt)
                    new_cnt_cur_ids.append(i)
                if best_id >= 0:
                    if best_id in particle_ids:
                        multiple_child_ids[i] = best_id
                    else:
                        particle_ids[i] = best_id
                        multiple_parent_ids[i] = list(set(potential_ids - ParticleTracker._INDEX_SHIFT) - {best_id})


        return new_countours, new_cnt_cur_ids, multiple_parent_ids, multiple_child_ids

    def mark_contours(self, img, contours=None):
        if contours is None:
            contours = copy.deepcopy(self.last_frame_contours)
        img_marked = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_marked, contours, contourIdx=-1, color=[0, 255, 0], thickness=2)
        return img_marked

    def save_checkpoint(self, basename="checkpoints/checkpoint"):
        fn = "{}_{}.json.bz2".format(basename, self.current_frame_number - 1)
        dirname = os.path.dirname(fn)
        if len(dirname) > 0:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        with bz2.open(fn, "wt") as f:
            json.dump(self.particles, f, indent=4)
        return fn

    def _fill_particle_ids_from_existing_contours(self, cur_img, cur_particle_ids, last_contours, last_particle_ids,
                                                  cur_contours, cur_cnt_ids):
        img_shape = cur_img.shape
        indexed_img = self._write_index_to_contours(img_shape, last_contours, last_particle_ids)
        new_contours, new_cnt_cur_ids, multiple_parent_ids, multiple_child_ids = self._fill_particle_ids_from_indexed_img(cur_particle_ids, cur_img,
                                                                                 cur_contours, cur_cnt_ids, indexed_img)
        return new_contours, new_cnt_cur_ids, multiple_parent_ids, multiple_child_ids

    def _get_historical_suspicious_contours_and_particle_ids(self):
        contours = []
        particle_ids = []
        for p_id, d in self.suspicious_contours.items():
            cnt = d[-1]["contour"]
            contours.append(cnt)
            particle_ids.append(p_id)
        return contours, particle_ids

    def track(self, img):
        print("INFO: current frame number", self.current_frame_number)
        _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)  # in case of JPG, information loss
        if self.last_img is None:
            self.rebase(img)
            self.current_frame_number += 1
            return

        cur_contours, cur_partilce_properies = self._find_image_contours_and_properties(img)
        cur_contours, cur_partilce_properies = zip(*sorted(zip(cur_contours, cur_partilce_properies),
                                                           key=lambda x: x[1][3], reverse=True))
        cur_particle_ids = [None] * len(cur_contours)
        # link particles to last frame
        new_contours, new_cnt_cur_ids, multiple_parent_ids, multiple_child_ids = \
            self._fill_particle_ids_from_existing_contours(img, cur_particle_ids, self.last_frame_contours,
            self.last_frame_particle_ids, cur_contours, list(range(len(cur_contours))))

        if len(multiple_parent_ids) > 0:
            print("Message: {} particles have multiple parents".format(len(multiple_parent_ids)))

        if len(multiple_child_ids) > 0:
            print("Message: {} particles have multiple children".format(len(set(multiple_child_ids.values()))))

        self.last_img = img
        pos_ids = [i for i in cur_particle_ids if i not in [None, -1]]
        assert len(pos_ids) == len(set(pos_ids))

        sus_candidate_contours = []
        sus_candidate_cnt_ids = []
        sus_guess_contours = []
        sus_guess_particle_ids = []
        for i_cnt, (i, pp) in enumerate(zip(cur_particle_ids, cur_partilce_properies)):
            if i not in [None, -1]:
                # (frame_number, circle center x , circle center y, area, centroid x, cnetroid y, gyradius)
                old_pp = self.particles[i][-1]
                old_position = np.array(old_pp[1:3])
                new_position = np.array(pp[1:3])
                old_rad = math.sqrt(old_pp[3]/math.pi)
                new_rad = math.sqrt(pp[3]/math.pi)
                particle_move = np.linalg.norm(new_position-old_position, ord=2)
                if particle_move > self.suspicious_move \
                        or math.fabs(new_rad - old_rad) > self.suspicious_rad_change \
                        or i_cnt in multiple_parent_ids or i_cnt in multiple_child_ids:
                    # the trajectory for this particle is suspicious
                    sus_candidate_contours.append(cur_contours[i_cnt])
                    sus_candidate_cnt_ids.append(i_cnt)
                    sus_guess_contours.append(cur_contours[i_cnt])
                    sus_guess_particle_ids.append(i)
                    # re-initiate detection placebo to link to the existing suspicious repository
                    cur_particle_ids[i_cnt] = None

        n_child_sus_added = 0
        for i_cnt, p_id in multiple_child_ids.items():
            if i_cnt not in sus_guess_particle_ids:
                assert cur_particle_ids[i_cnt] is None
                sus_candidate_contours.append(cur_contours[i_cnt])
                sus_candidate_cnt_ids.append(i_cnt)
                sus_guess_contours.append(cur_contours[i_cnt])
                if len(self.suspicious_contours.keys()) > 0:
                    huge_p_id = max(max(self.suspicious_contours.keys()), self.MAX_NUM_PARTICLES) + \
                                n_child_sus_added + 1
                else:
                    huge_p_id = self.MAX_NUM_PARTICLES + n_child_sus_added + 1
                sus_guess_particle_ids.append(huge_p_id)
                n_child_sus_added += 1


        if len(sus_candidate_cnt_ids) > 0:
            print("Message: {} particles are suspicious".format(len(sus_candidate_cnt_ids)))

        sure_particle_ids = copy.deepcopy(cur_particle_ids)


        # link the suspicious particles to the suspicious memory
        sus_hist_contours, sus_hist_particle_ids = self._get_historical_suspicious_contours_and_particle_ids()
        # existing suspicious particles
        new_sus_candidate_contours, new_sus_candidate_cnt_ids, _, _ = self._fill_particle_ids_from_existing_contours(img,
                cur_particle_ids, sus_hist_contours, sus_hist_particle_ids, sus_candidate_contours,
                sus_candidate_cnt_ids)
        # new suspicious particles
        new_sus_candidate_contours, new_sus_candidate_cnt_ids, _, _ = self._fill_particle_ids_from_existing_contours(img,
                cur_particle_ids, sus_guess_contours, sus_guess_particle_ids, new_sus_candidate_contours,
                new_sus_candidate_cnt_ids)

        assert len(new_sus_candidate_contours) == 0
        assert len(new_sus_candidate_cnt_ids) == 0
        assert len(sus_candidate_cnt_ids) + len([i for i in sure_particle_ids if i not in [None, -1]]) == \
               len([i for i in cur_particle_ids if i not in [None, -1]])

        assert -1 not in cur_particle_ids

        n_new_particles = 0
        n_skipped_suspicious = 0
        for i_cnt, (cnt, p_id, prop) in enumerate(zip(cur_contours, cur_particle_ids, cur_partilce_properies)):
            if p_id is None:
                # New particle
                center = (prop[1], prop[2])
                if self._center_inside_image(center, self.enter_margin, img):
                    # large enough particle
                    p_id = len(self.particles)
                    self.particles.append(list())
                    cur_particle_ids[i_cnt] = p_id
                    self.particles[p_id].append(prop)
                    n_new_particles += 1
                else:
                    cur_particle_ids[i_cnt] = -1
            else:
                # Existing particles
                if p_id < self.MAX_NUM_PARTICLES:
                    assert p_id != -1
                    if p_id in sure_particle_ids:
                        self.particles[p_id].append(prop)
                else:
                    n_skipped_suspicious += 1
        if n_new_particles > 0:
            print("Message: {} new particles detected".format(n_new_particles))
        if n_skipped_suspicious > 0:
            print("Message: {} suspicious new particles are skipped".format(n_skipped_suspicious))

        assert None not in cur_particle_ids

        for i_cnt in sus_candidate_cnt_ids:
            p_id = cur_particle_ids[i_cnt]
            cnt = cur_contours[i_cnt]
            self.suspicious_contours[p_id].append({"contour": cnt, "prop": cur_partilce_properies[i_cnt]})

        unstable_suspicious_particle_ids = set(sure_particle_ids) & set(self.suspicious_contours.keys())
        for p_id, sus_traj in self.suspicious_contours.items():
            if self.current_frame_number - sus_traj[-1]["prop"][0] >= self.suspicious_particle_max_missing_frames:
                # the suspicious is not stable, discard it
                unstable_suspicious_particle_ids.add(p_id)
        for p_id in unstable_suspicious_particle_ids:
            self.suspicious_contours.pop(p_id)
        if len(unstable_suspicious_particle_ids) > 0:
            print("Message: {} suspicious particles are unstable and discarded from suspicious memory"
                  .format(len(unstable_suspicious_particle_ids)))

        num_suspicious_particles_confirmed = 0
        confirmed_suspicious_particle_ids = []
        for p_id, sus_traj in self.suspicious_contours.items():
            if len(sus_traj) >= self.suspicious_confirm_frames:
                confirmed_suspicious_particle_ids.append(p_id)  # remove from suspicious queue
                traj_pps = [d["prop"] for d in sus_traj]
                new_p_id = p_id
                if new_p_id > self.MAX_NUM_PARTICLES:
                    new_p_id = len(self.particles)
                    # this suspicious one is a new particle
                    self.particles.append(list())
                self.particles[new_p_id].extend(traj_pps)
                i_cnt = cur_particle_ids.index(p_id)
                sure_particle_ids[i_cnt] = new_p_id
                cur_particle_ids[i_cnt] = new_p_id
                num_suspicious_particles_confirmed += 1
            else:
                if p_id in cur_particle_ids and sus_traj[-1]["prop"][0] == self.current_frame_number:
                    i_cnt = cur_particle_ids.index(p_id)
                    cur_particle_ids[i_cnt] = None
                else:
                    raise ValueError("Shouldn't reach here")
        for p_id in confirmed_suspicious_particle_ids:
            self.suspicious_contours.pop(p_id)

        if num_suspicious_particles_confirmed > 0:
            print("Message: {} suspicious particles are confirmed to be real".format(
                num_suspicious_particles_confirmed))

        # make a copy of current frame to be compared in next frame
        last_2nd_frame_particle_ids = copy.deepcopy(self.last_frame_particle_ids)
        last_2nd_frame_contours = copy.deepcopy(self.last_frame_contours)
        last_2nd_frame_properties = copy.deepcopy(self.last_frame_properties)
        self.last_frame_particle_ids = copy.deepcopy([i for i in cur_particle_ids
                                                      if i not in [None, -1]])
        self.last_frame_contours = copy.deepcopy([cnt for i_cnt, cnt in enumerate(cur_contours)
                                                  if cur_particle_ids[i_cnt] not in [None, -1]])
        # None: not assign, -1: out of boundary new particles
        self.last_frame_properties = copy.deepcopy([pp for i_cnt, pp in enumerate(cur_partilce_properies)
                                                    if cur_particle_ids[i_cnt] not in [None, -1]])
        last_2nd_coords = np.array([prop[1:3] for prop in last_2nd_frame_properties])
        last_2nd_area_list = np.array([prop[3] for prop in last_2nd_frame_properties])
        last_2nd_rad_list = np.sqrt(last_2nd_area_list / math.pi)
        last_1st_coords = np.array([prop[1:3] for prop in self.last_frame_properties])
        last_1st_area_list = np.array([prop[3] for prop in self.last_frame_properties])
        last_1st_rad_list = np.sqrt(last_1st_area_list / math.pi)
        last_2nd_to_1st_distance = distance_matrix(last_2nd_coords, last_1st_coords)

        # add particle history to the frame memory
        num_too_old_particles = 0
        num_too_close_particles = 0
        num_new_history_particles = 0
        history_candidate_particle_ids = set(last_2nd_frame_particle_ids) - set(self.last_frame_particle_ids)
        for p_id in history_candidate_particle_ids:
            # skip very old particles
            if self.current_frame_number - self.particles[p_id][-1][0] <= self.max_history_frame_num:
                last_2nd_i_cnt = last_2nd_frame_particle_ids.index(p_id)
                dist = last_2nd_to_1st_distance[last_2nd_i_cnt, :] - \
                       (last_2nd_rad_list[last_2nd_i_cnt] + last_1st_rad_list)
                assert dist.shape == last_1st_rad_list.shape
                assert len(dist.shape) == 1
                dist = dist.min()
                if dist > self.min_gap:
                    # the particle is a separated one from the current frame
                    self.last_frame_particle_ids.append(p_id)
                    self.last_frame_contours.append(last_2nd_frame_contours[last_2nd_i_cnt])
                    self.last_frame_properties.append(last_2nd_frame_properties[last_2nd_i_cnt])
                    num_new_history_particles += 1
                else:
                    num_too_close_particles += 1
            else:
                num_too_old_particles += 1

        assert num_too_close_particles + num_too_old_particles + num_new_history_particles \
            == len(history_candidate_particle_ids)

        if num_too_old_particles + num_too_close_particles > 0:
            print("Message: {} particles are removed from history, in which {} particles are too old, {} particles "
                  "are too close to the particle in the current frame".format(
                       num_too_old_particles + num_too_close_particles, num_too_old_particles,
                       num_too_close_particles))
        if num_new_history_particles > 0:
            print("Message: {} particles in history now".format(num_new_history_particles))
        if len(self.suspicious_contours) > 0:
            print("Message: {} particles in suspicious memory now".format(len(self.suspicious_contours)))
        print()

        pos_ids = [i for i in cur_particle_ids if i not in [None, -1]]
        assert len(pos_ids) == len(set(pos_ids))

        self.current_frame_number += 1

