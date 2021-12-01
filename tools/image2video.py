import os
import subprocess


def image2video(forder):
    subprocess.call(["ffmpeg", "-framerate", "10", "-i", "step%03d.png", "video.mp4"])


def image2video_with_num(forder, ite, num):
    os.chdir(forder)
    subprocess.call(["ffmpeg", "-framerate", "10", "-i", "step%03d.png", f"../ite{ite}_video{num}.mp4"])


if __name__ == "__main__":
    # image2video("results/HorizonCrossing-v0/experiment-2021-11-07-21-09-43/logs/tester/test-2021-11-08-10-33-39/episode4")
    name = "C"
    # name = "M"
    experiment_folder = "experiment-2021-11-24-13-13-05"
    test_forder = "test-2021-11-25-10-48-08"
    iterations = 299000
    episode_num = 5

    ENV_name = "HorizonCrossing-v0" if name == "C" else "HorizonMultilane-v0"
    root_folder = os.path.join("./results", ENV_name, experiment_folder, "logs/tester", test_forder)

    root_folder = os.path.abspath(root_folder)

    for i in range(episode_num):
        image_forder = os.path.join(root_folder, "ite{}_episode{}".format(iterations, i))
        image2video_with_num(image_forder, iterations, i)
