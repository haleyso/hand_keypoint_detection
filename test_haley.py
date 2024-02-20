# works w openmmlab2 
from mmpose.apis import MMPoseInferencer

# build the inferencer with 2d model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose2d='configs/hand_2d_keypoint/rtmpose/hand5/no_depthwise_relu_nochannelwiseatn_teensy.py',
    pose2d_weights='work_dirs/no_depthwise_relu_nochannelwiseatn_teensy/haley.pth',
    scope='mmpose'
)

img_path = '/home/haleyso/mmpose/data/onehand10k/Test/source/1491.jpg'
# folder_path = 'data/onehand10k/Test/source/'
# result_generator = inferencer(folder_path, show=True)
# results = [result for result in result_generator]

result_generator = inferencer(img_path, vis_out_dir='vis_results')
result = next(result_generator)

# print(result)