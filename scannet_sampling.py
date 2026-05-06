import numpy as np
import torch
from collections import defaultdict
import math
import os

import time


class SCANNET_MY_DOWN_SAMPLING():
    def __init__(self):
        self.object_class  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.big_background = [
    0, 
    1
]
        self.total_time = 0

    def get_instance_dict(self,pt_root,scene):
        pt_file_path = os.path.join(pt_root,scene+"_outputs.pt")

        self.data = torch.load(pt_file_path, map_location='cpu')
        self.instance_out = self.data  
        self.instance_mask = self.instance_out['pts_instance_mask'][1] 
        self.segmentic_mask = self.instance_out['pts_semantic_mask'][1]
        self.total_num_points = len(self.segmentic_mask)

        instance_dict = defaultdict(list)
        non_object_dict = defaultdict(list)
        start_time = time.time()

        object_mask = np.isin(self.segmentic_mask, self.object_class)
        object_indices = np.where(object_mask)[0]
        object_instance_ids = self.instance_mask[object_indices]

        uids, inv = np.unique(object_instance_ids, return_inverse=True)
        for i, ins_id in enumerate(uids):
            instance_dict[int(ins_id)] = object_indices[inv == i]

        # non-object
        big_bg_mask = np.isin(self.segmentic_mask, self.big_background)
        non_object_mask = ~object_mask

        # big background
        bg_mask = non_object_mask & big_bg_mask
        bg_indices = np.where(bg_mask)[0]
        bg_instance_ids = self.instance_mask[bg_indices]

        uids, inv = np.unique(bg_instance_ids, return_inverse=True)
        for i, ins_id in enumerate(uids):
            non_object_dict[int(ins_id)] = bg_indices[inv == i]


        non_object_dict[-1] = np.where(non_object_mask & (~big_bg_mask))[0]

        self.total_time += (time.time() - start_time)

        print(len(non_object_dict[-1]))

        return instance_dict,non_object_dict

    def object_and_non_object_sampling(self,pt_root,scene, num_sampling):
        instance_dict,non_object_dict = self.get_instance_dict(pt_root,scene)

        start_time = time.time()

        if self.total_num_points <= num_sampling:
            print("total num point is small than num_sampling, so don't have to sampling")
            print(pt_root)
            return instance_dict,non_object_dict

        non_object_num_points = sum(len(v) for v in non_object_dict.values())
        object_num_points = self.total_num_points - non_object_num_points


        object_ratio = object_num_points/self.total_num_points
        non_object_ratio = 1 - object_ratio
        object_samping_ratio = object_num_points/num_sampling

        w =  math.exp(-2*object_ratio)
    
        using_object_ratio = (object_ratio * w + non_object_ratio*(1-w))

        using_object_ratio = max(using_object_ratio,object_ratio)


        if using_object_ratio >= object_samping_ratio:
            using_object_ratio = object_samping_ratio
        
        else:
            if using_object_ratio *num_sampling + non_object_num_points < num_sampling:
                using_object_ratio = object_samping_ratio

        print("object_points_ratio: ",object_ratio)
        print("object_samping_ratio: ",object_samping_ratio)
        print("using_object_points_ratio: ",using_object_ratio)

        using_object_num_points = using_object_ratio*num_sampling

        print("using_object_num_points; ", using_object_num_points)

        total_num_object = 0
        for key, value in instance_dict.items():
            if key != -1:
                object_sampling_ratio = using_object_num_points/object_num_points
                num_each_key_object = len(value)
                if num_each_key_object < math.ceil(num_each_key_object*object_sampling_ratio):
                    object_sampling_index = np.random.choice(value,size=num_each_key_object,replace=False)
                else:
                    object_sampling_index = np.random.choice(value,size=math.ceil(num_each_key_object*object_sampling_ratio),replace=False)

                instance_dict[key] = object_sampling_index

                total_num_object += len(object_sampling_index)

        print("sample_total_num_object: ",total_num_object)

        using_non_object_num_points = num_sampling-total_num_object
        print("sample_total_num_non_object: ",using_non_object_num_points)

        total_non_object_num_points = 0 

        non_object_sampling_ratio = using_non_object_num_points/non_object_num_points

        num_each_key_object = len(non_object_dict[0])
        non_object_sampling_index = np.random.choice(non_object_dict[0],size=math.ceil(num_each_key_object*non_object_sampling_ratio),replace=False)
        non_object_dict[0] = non_object_sampling_index
   
        total_non_object_num_points += len(non_object_sampling_index)
        non_object_dict[1] = np.random.choice(non_object_dict[1],size=num_sampling - (total_num_object + total_non_object_num_points) ,replace=False)

        self.total_time += (time.time() - start_time)

        return instance_dict,non_object_dict 

def my_down_sampling(my_downsampling_index_result,scene,data_root,output_root):
    POINT_DTYPE = np.float32
    INSTANCE_DTYPE = np.int64
    SEMANTIC_DTYPE = np.int64
    SUPERPOINT_DTYPE = np.int64
    POINT_DIM = 6 


    points_dir = os.path.join(data_root, "points")
    instance_dir = os.path.join(data_root, "instance_mask")
    semantic_dir = os.path.join(data_root, "semantic_mask")
    superpoint_dir = os.path.join(data_root, "super_points_spt") #

    out_points_dir = os.path.join(output_root, "points")
    out_instance_dir = os.path.join(output_root, "instance_mask")
    out_semantic_dir = os.path.join(output_root, "semantic_mask")
    out_superpoint_dir = os.path.join(output_root, "super_points_spt") #
    


    os.makedirs(out_points_dir, exist_ok=True)
    os.makedirs(out_instance_dir, exist_ok=True)
    os.makedirs(out_semantic_dir, exist_ok=True)
    os.makedirs(out_superpoint_dir, exist_ok=True)

    indices = my_downsampling_index_result

    bin_name = scene + ".bin"
    print(f"Processing: {bin_name}")

    # ---------- load ----------
    points = np.fromfile(
        os.path.join(points_dir, bin_name),
        dtype=POINT_DTYPE
    ).reshape(-1, POINT_DIM)

    instance_mask = np.fromfile(
        os.path.join(instance_dir, bin_name),
        dtype=INSTANCE_DTYPE
    )

    semantic_mask = np.fromfile(
        os.path.join(semantic_dir, bin_name),
        dtype=SEMANTIC_DTYPE
    )

    super_points = np.fromfile(
        os.path.join(superpoint_dir, bin_name),
        dtype=SUPERPOINT_DTYPE
    )
    indices = indices.astype(np.int64)
    start = time.time()

    points_sel = points[indices]
    instance_sel = instance_mask[indices]
    semantic_sel = semantic_mask[indices]
    superpoint_sel = super_points[indices]

    end = time.time()
    save_time = end - start
    if len(points) == len(super_points): 
        pass
    else:
        print("!!!wrong len!!!")

    points_sel.tofile(os.path.join(out_points_dir, bin_name))
    instance_sel.tofile(os.path.join(out_instance_dir, bin_name))
    semantic_sel.tofile(os.path.join(out_semantic_dir, bin_name))
    superpoint_sel.tofile(os.path.join(out_superpoint_dir, bin_name))
    print(f"Saved {len(indices)} points for {scene}")
    print("=======================================\n")
    
    return save_time
DOWN_SAMPLING = SCANNET_MY_DOWN_SAMPLING()

scene_list_txt = "/raid2/jyoung/val_scene_names.txt"

data_root = "/raid2/jyoung/data/scannet" 
output_root = "/raid2/jyoung/else/sampling_100K/scannet" 

pt_root = "/raid2/jyoung/vdg-uni3dseg_results/scannet" 


with open(scene_list_txt, "r") as f:
    scenes = [line.strip() for line in f if line.strip()]

print(f"Total scenes: {len(scenes)}")

num_sampling = 100000

total_save_time = 0
for scene in scenes:
    # if os.path.exists(os.path.join(output_root, "points",scene+".bin")):
    #     print(f"{scene}.bin already exists. Skipping...")
    #     continue 
    pt_path = os.path.join(pt_root,scene+"_outputs.pt")
    downsampled_istance_dict,non_object_dict = DOWN_SAMPLING.object_and_non_object_sampling(pt_root,scene,num_sampling)


    a = time.time()
    all_arrays = list(downsampled_istance_dict.values()) + list(non_object_dict.values())
    

    my_downsampling_index_result = np.sort(np.concatenate(all_arrays))
    b = time.time()-a
    print(scene)
    total_save_time +=b
    save_time = my_down_sampling(my_downsampling_index_result,scene,data_root,output_root)
    total_save_time += save_time

print(DOWN_SAMPLING.total_time)
print(total_save_time)

print(total_save_time + DOWN_SAMPLING.total_time)