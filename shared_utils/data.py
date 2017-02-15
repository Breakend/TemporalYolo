"""
A class for loading data, in particular loading YOLO features in batches
"""
import os


def load_regular_coord_by_line(line):
    """
    Note this is borrowed from original rolo code
    """
    elems = line.split('\t')
    if len(elems) < 4:
        elems = line.split(',')
        if len(elems) < 4:
            elems = line.split(' ')

    [X1, Y1, W, H] = elems[0:4]
    coord_regular = [int(X1), int(Y1), int(W), int(H)]
    return coord_regular


def coord_regular_to_decimal(coord_regular, img_wid, img_ht):
    """
    Note this is borrowed from original rolo code
    """
    img_wid *= 1.0
    img_ht *= 1.0
    coord_decimal = list(coord_regular)

    # convert top-left point (x,y) to mid point (x, y)
    coord_decimal[0] += coord_regular[2] / 2.0
    coord_decimal[1] += coord_regular[3] / 2.0

    # convert to [0, 1]
    coord_decimal[0] /= img_wid
    coord_decimal[1] /= img_ht
    coord_decimal[2] /= img_wid
    coord_decimal[3] /= img_ht

    return coord_decimal




class BatchLoader:

    def __init__(self, data_filepath, seq_len=6, batch_size=1, step_size=5, folders_to_use=None):
        self.batches = self.generate_batches(data_filepath, seq_len, batch_size, folders_to_use, step_size)

    def load_batch(self, batch_id):
        batch = self.batches[batch_id]
        batch_xs = []
        batch_ys = []
        vec_len = 4102
        for ground_truth_filepath, frames, frame_ids in batch:
            import pdb; pdb.set_trace()
            with open(ground_truth_filepath) as gt_file:
                lines = gt_file.readlines()
                for x in frame_ids:
                    #frame ids are 1 indexed
                    reg_coords = load_regular_coord_by_line(lines[x-1])
                    decimal_coords = coord_regular_to_decimal(reg_coords, img_wid, img_ht)
                    batch_ys.append(decimal_coords)

            for frame in frames:
                vec_from_file = np.load(file_path)
                vec_from_file = np.flatten(vec_from_file)
                vec_len = vec_from_file.shape[0]
                # TODO: may need to remove category/etc
                batch_xs.append(vec_from_file)

        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)
        batch_xs = np.reshape(batch_xs, [len(frames) * len(batch), vec_len])
        batch_ys = np.reshape(batch_ys, [len(frames) * len(batch), vec_len])

        batch_xs, batch_ys


    def generate_batches(self, data_filepath, seq_len=6, batch_size=1, folders_to_use=None, step_size=5):
        """Expects a folder structure in the format:
           -data_filepath
             -> folders_to_use[0]
              -> groundtruth_rect.txt (list of N boxes of length 4)
              -> yolo_output
                 -> 0001.npy (a 4102 feature vector which is the YOLO features for frame 1)
                 -> ...
                 -> NNNN.npy (a 4102 feature vector which is the YOLO features for frame N)
             -> folders_to_use[1]
             -> folders_to_use[2]

            folders_to_use = a list of folders which contain data as seen above (aka ["Birds1", "Basketball"])
            seq_len = the number of steps in sequence
            step_size = the number of frames to skip for timestep (i.e. step_size=1, is just the normal video)

            returns batches with references to data to load in the format: [   ( ground_truth_filepath,[frame_paths],[frame_ids])    ]
            in this way frame_paths refers to the .npy yolo features and frame_ids refers to the indices into
            the groundtruth file for the bounding box coordinates so we can index into it to get the frame ground truths for a batch
        """
        batches = []
        if not folders_to_use:
            raise Exception("TODO: default to listing directories, but for now need to pass a list of directories")

        frames_per_folder = {}
        # import pdb; pdb.set_trace()
        for f in folders_to_use:
            frames_per_folder[f] = [ fi for fi in os.listdir(os.path.join(os.path.join(data_filepath, f), 'yolo_out/')) if fi.endswith('.npy') ]

        current_step = 1
        possible_batches = []
        while True:
            failure_count = 0
            for f in folders_to_use:
                # make sure all the frames we want are in this batch
                all_frames = all(["%04d.npy" % x in frames_per_folder[f] for x in range(current_step, current_step + seq_len)])

                # if we can't find all the frames for a given folder, give up on this batch
                if not all_frames:
                    failure_count += 1
                    continue

                frames = [os.path.join(*[data_filepath, f, 'yolo_out/',"%04d.npy" % x]) for x in range(current_step, current_step + seq_len)]
                frame_ids = [x for x in range(current_step, current_step + seq_len)]
                # import pdb; pdb.set_trace()
                # TODO: load sample image to collect image height and width data
                possible_batches.append((os.path.join(*[data_filepath, f, "groundtruth_rect.txt"]), frames, frame_ids))

            if failure_count >= len(folders_to_use):
                # all our folders are out of possible sequences
                break

            current_step += step_size

        batches = [possible_batches[x:x+batch_size] for x in xrange(0, len(possible_batches), batch_size)]
        # print(batches)

        return batches
