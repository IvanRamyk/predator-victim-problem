import cv2

import tensorflow as tf
import numpy as np

def evaluate(trainer, PVEnv, n_iter=10, video_file=None):
    if video_file is not None:
        video = cv2.VideoWriter("../videos/Predator_Victim.avi", 0, 60, (PVEnv.screen_wh, PVEnv.screen_wh))
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        while not done:
            action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            action_victim = trainer.compute_action(obs['victim'], policy_id="policy_victim", explore=False)
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victim": action_victim})
            done = dones['__all__']
            frame = PVEnv.render(mode='rgb_array')
            if video_file is not None:
                video.write(frame[..., ::-1])
        PVEnv.close()
    if video_file is not None:
        video.release()


def evaluate_length(trainer, PVEnv, file, n_iter=100):
    f = open(file, "w")
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        step = 0
        while not done:
            action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            action_victim = trainer.compute_action(obs['victim'], policy_id="policy_victim", explore=False)
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victim": action_victim})
            done = dones['__all__']
            step += 1
        PVEnv.close()
        f.write("{} {}\n".format(i, step))

    f.close()
        



def victim_custom(trainer, PVEnv, n_iter=10, video_file=None):
    if video_file is not None:
        video = cv2.VideoWriter("../videos/Predator_Victim.avi", 0, 60, (PVEnv.screen_wh, PVEnv.screen_wh))
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        while not done:
            action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            print(action_predator)
            # action_victim = trainer.compute_action(obs['victim'], policy_id="policy_victim", explore=False)
            try:
                a_x = float(input())
            except:
                a_x = 0
            try:
                a_y = float(input())
            except:
                a_y = 0
            action_victim = np.array([a_x, a_y])
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victim": action_victim})
            done = dones['__all__']
            frame = PVEnv.render(mode='rgb_array')
            if video_file is not None:
                video.write(frame[..., ::-1])
        PVEnv.close()
    if video_file is not None:
        video.release()
