"""
A class for loading data, in particular loading YOLO features in batches
"""

class BatchLoader:



    def generate_batches(data_filepath, seq_size=6, folders_to_use=None):
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

            returns batches with references to data to load in the format: [   (  [frame_paths],[frame_ids])    ]
            in this way frame_paths refers to the .npy yolo features and frame_ids refers to the indices into
            the groundtruth file for the bounding box coordinates
        """
        batches = []

        for f in folders_to_use:



        return batches
