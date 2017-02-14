"""
A class for loading data, in particular loading YOLO features in batches
"""

class BatchLoader:

    def __init__(data_filepath, seq_size=6, batch_size=1, folders_to_use=None):
        self.batches = generate_batches(data_filepath, seq_size, batch_size, folders_to_use)

    def load_batch(batch_id):
        #TODO:

    def generate_batches(data_filepath, seq_size=6, batch_size=1, folders_to_use=None):
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
            seq_size = the number of steps in sequence
            step_size = the number of frames to skip for timestep (i.e. step_size=1, is just the normal video)

            returns batches with references to data to load in the format: [   ( ground_truth_filepath,[frame_paths],[frame_ids])    ]
            in this way frame_paths refers to the .npy yolo features and frame_ids refers to the indices into
            the groundtruth file for the bounding box coordinates so we can index into it to get the frame ground truths for a batch
        """
        batches = []
        if not folders_to_use:
            raise Exception("TODO: default to listing directories, but for now need to pass a list of directories")

        frames_per_folder = {}
        # TODO: populate


        #TODO: make step_size variable so that we don't just offset the training batches by a few frames
        step_size = 5
        current_step = 0
        possible_batches = []
        while True:
            failure_count = 0
            for f in folders_to_use:
                # make sure all the frames we want are in this batch
                all_frames = ["%04d.npy" % x in frames_per_folder[f] for x in range(current_step, current_step + seq_size)].all()

                # if we can't find all the frames for a given folder, give up on this batch
                if not all_frames:
                    failure_count += 1
                    continue

                frames = ["%04d.npy" % x for x in range(current_step, current_step + seq_size)]
                frame_ids = [x for x in range(current_step, current_step + seq_size)]
                possible_batches.append(("%s/groundtruth_rect.txt" % f, frames, frame_ids))

            if failure_count >= len(folders_to_use):
                # all our folders are out of possible sequences
                break
                
            current_step += step_size

        batches = [possible_batches[x:x+batch_size] for x in xrange(0, len(data), batch_size)]

        return batches
