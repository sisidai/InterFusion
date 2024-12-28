import argparse
import os
import torch
import numpy as np
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from sklearn.cluster import KMeans
import time
from utils import load_pose, build_codebook


def main(in_path, out_path, cluster_size):

    vp_path = '../data/vposer'
    vp, _ = load_model(vp_path, model_code=VPoser,
                    remove_words_in_model_weights='vp_model.',
                    disable_grad=True)

    t0 = time.time()

    # get pose embeddings
    pose_embeddings = []
    names = os.listdir(in_path)
    for n in range(len(names)):
        pose = load_pose(os.path.join(in_path, names[n]))
        pose_embedding = vp.encode(pose).mean
        pose_embeddings.append(pose_embedding.detach().numpy())

    pose_embeddings = np.concatenate(pose_embeddings, axis = 0)
    print(f"[INFO] Totally {len(names)} poses loaded and embedded")

    # get clusters by KMeans
    np.random.seed(seed=0)
    kmeans = KMeans(n_clusters=cluster_size, random_state=0, n_init='auto').fit(pose_embeddings)
    centroids = kmeans.cluster_centers_
    centroids = torch.from_numpy(centroids).type(torch.float)
    clustered_poses = vp.decode(centroids)['pose_body'].reshape(cluster_size, -1)
    print(f"[INFO] Clustering finished: {clustered_poses.shape}")

    build_codebook(clustered_poses, out_path)

    t1 = time.time()
    print(f"[INFO] Total time used: {t1 - t0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_path', type=str, default='results/estimated_poses')
    parser.add_argument('-out_path', type=str, default='results/codebook.pth')
    parser.add_argument('-cluster_size', type=int, default=2048)

    args = parser.parse_args()

    main(args.in_path, args.out_path, args.cluster_size)
