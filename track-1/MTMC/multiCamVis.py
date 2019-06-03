from moviepy.editor import VideoFileClip, CompositeVideoClip

TRACKLET = 'deep_sort'
eval_dir = 'eval_0528'
video_root_path = "res/%s/MCT/%s" % (TRACKLET, eval_dir)

clip0 = VideoFileClip(video_root_path + '/' + "S2c06_MTMC.avi").resize(0.5)
clip0.set_start(0)
clip1 = VideoFileClip(video_root_path + '/' + "S2c07_MTMC.avi").resize(0.5)
clip1.set_start(0.061)
clip2 = VideoFileClip(video_root_path + '/' + "S2c08_MTMC.avi").resize(0.5)
clip2.set_start(0.421)
clip3 = VideoFileClip(video_root_path + '/' + "S2c09_MTMC.avi").resize(0.5)
clip3.set_start(0.660)

final_clip = CompositeVideoClip([clip0,clip1.set_pos((960,0)),clip2.set_pos((0,540)),clip3.set_pos((960,540))],size=(1920,1080))
final_clip.write_videofile("%s/S2_multi_cam.mp4" % video_root_path)

