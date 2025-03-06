# -*- coding:utf8 -*-
import cv2
import os
import shutil


def get_frame_from_video(_video_name, _save_path):
    """
    :param _video_name: 输入视频路径
    :param num_s: 保存图片的帧率间隔
    :param _save_path: 抽出的图片保存的位置
    :param _txt_path: 记录图片路径和名称的txt文件路径
    """
    # 获取视频文件名
    file_name = os.path.basename(_video_name).split('.')[0]
    #
    # # 创建以视频名字命名的文件夹
    # video_save_path = os.path.join(_save_path, file_name)
    # is_exists = os.path.exists(video_save_path)
    # if not is_exists:
    #     os.makedirs(video_save_path)
    #     print('path of %s is build' % video_save_path)
    #
    #     # 开始读视频
    #     video_capture = cv2.VideoCapture(_video_name)
    #     fps = video_capture.get(5)
    #     i = 0
    #     total_frames = 0
    #
    #     while True:
    #         success, frame = video_capture.read()
    #         if not success:
    #             break
    #         total_frames += 1
    #
    #     video_capture.release()
    #     video_capture = cv2.VideoCapture(_video_name)
    #     i = 0
    #     j = 0
    #
    #     with open(_txt_path, 'a') as txt_file:
    #         while True:
    #             success, frame = video_capture.read()
    #             i += 1
    #             if i % int(fps / num_s) == 0:
    #                 # 保存图片
    #                 try:
    #                     j += 1
    #                     save_name = f"img_{j:05d}.png"
    #                     cv2.imwrite(os.path.join(video_save_path, save_name), frame)
    #                 except:
    #                     print('An unknown error occurred! Skipping')
    #
    #                 print('image of %s is saved' % save_name)
    #             if not success:
    #                 print('video is all read')
    #                 break
    #
    #         # 将视频文件夹路径、总帧数和标签写入txt文件
    #         txt_file.write(video_save_path + ' ' + str(total_frames) + ' 2\n')


      # 获取视频总帧数
    video_capture = cv2.VideoCapture(_video_name)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {file_name}, Total frames: {total_frames}")

    folder_index = 0
    frame_index = 0

    while frame_index < total_frames:
        # 创建文件夹，直接保存在 _save_path 下
        folder_name = f"{file_name}_{folder_index:03d}"
        folder_path = os.path.join(_save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

        # 读取并保存帧
        img = 1  # 每个文件夹内图片编号从 00001 开始
        for i in range(128):
            success, frame = video_capture.read()
            if not success:
                print("Reached end of video.")
                break
            save_name = f"img_{img:05d}.png"
            cv2.imwrite(os.path.join(folder_path, save_name), frame)
            img += 1
            frame_index += 1

        folder_index += 1

        # 如果读取到视频末尾，退出循环
        if not success:
            print(f"Finished processing video: {file_name}.")
            break

    video_capture.release()
    print(f"Finished processing video: {file_name}. Total folders created: {folder_index}")


if __name__ == '__main__':
    # 视频文件名字

    file_path = '/home/heyuxin/anaconda3/envs/pytorch/SDA-CLIP-main/Dataset/videos/test1/'
    save_path = '/mnt/sdc/heyuxin/surgvisdom_image/test/'
    # txt_path = '/home/heyuxin/anaconda3/envs/pytorch/SDA-CLIP-main/lists/surgvisdom/val_frames.txt'
    files = os.listdir(file_path)  # 采用listdir来读取所有文件
    files.sort()
    # interval = 20  # 设置每秒抽多少帧
    for file_ in files:
        video_name = os.path.join(file_path, file_)
        get_frame_from_video(video_name, save_path)