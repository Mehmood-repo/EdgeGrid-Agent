"""Spatio-Temporal Graph Topology Generator for Wind Turbine Networks.

Constructs static topological and geometric graphs (Delaunay triangulation,
k-NN, and distance-thresholded) from physical turbine coordinates.
"""

from __future__ import annotations

import os
from typing import Literal, Tuple, Union, Optional
import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data


class WindTurbineGraph:
    """Manages spatial coordinates, distance matrices, and graph topologies

    for a wind turbine farm.
    """

    def __init__(
        self,
        locations: Union[str, os.PathLike, pd.DataFrame, np.ndarray],
        method: Literal["delaunay", "knn", "threshold", "hybrid"] = "delaunay",
        knn_k: int = 6,
        dist_threshold: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> None:
        """Initialize the turbine graph topology.

        Args:
            locations: Path to CSV, pandas DataFrame, or numpy array of coordinates.
                       If CSV or DataFrame, expects columns ['TurbID', 'x', 'y'].
            method: Graph construction algorithm ('delaunay', 'knn', 'threshold', 'hybrid').
            knn_k: Number of nearest neighbors to connect if method is 'knn' or 'hybrid'.
            dist_threshold: Maximum physical Euclidean distance (meters) for edge connectivity.
            sigma: Scaling parameter for Gaussian distance kernel. If None, uses median edge distance.
        """
        self.method = method
        self.knn_k = knn_k
        self.dist_threshold = dist_threshold

        self.turb_ids, self.coords = self._parse_locations(locations)
        self.num_nodes = len(self.turb_ids)
        self.id_to_idx = {turb_id: idx for idx, turb_id in enumerate(self.turb_ids)}
        self.idx_to_id = {idx: turb_id for idx, turb_id in enumerate(self.turb_ids)}

        # Pairwise Euclidean distances and directional compass bearings
        self.dist_matrix = self._compute_distance_matrix()
        self.bearing_matrix = self._compute_bearing_matrix()

        # Build graph edges
        self.edge_index_np, self.edge_dist_np, self.edge_bearing_np = self._build_edges()

        # Gaussian kernel parameter
        if sigma is None:
            self.sigma = float(np.median(self.edge_dist_np)) if len(self.edge_dist_np) > 0 else 1000.0
        else:
            self.sigma = float(sigma)

        self.edge_weight_np = np.exp(-((self.edge_dist_np / self.sigma) ** 2))

        # Convert to PyTorch tensors
        self.edge_index = torch.from_numpy(self.edge_index_np).long()
        self.edge_dist = torch.from_numpy(self.edge_dist_np).float()
        self.edge_bearing = torch.from_numpy(self.edge_bearing_np).float()
        self.edge_weight = torch.from_numpy(self.edge_weight_np).float()
        self.pos = torch.from_numpy(self.coords).float()

    def _parse_locations(
        self, locations: Union[str, os.PathLike, pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parses coordinates into ordered turbine IDs and (x, y) coordinates."""
        if isinstance(locations, (str, os.PathLike)):
            df = pd.read_csv(locations)
        elif isinstance(locations, pd.DataFrame):
            df = locations.copy()
        elif isinstance(locations, np.ndarray):
            num_nodes = locations.shape[0]
            turb_ids = np.arange(1, num_nodes + 1)
            return turb_ids, locations.astype(np.float32)
        else:
            raise TypeError(f"Unsupported locations format: {type(locations)}")

        # Ensure sorted by TurbID for deterministic node indexing
        df = df.sort_values(by="TurbID").reset_index(drop=True)
        turb_ids = df["TurbID"].values
        coords = df[["x", "y"]].values.astype(np.float32)
        return turb_ids, coords

    def _compute_distance_matrix(self) -> np.ndarray:
        """Computes symmetric pairwise Euclidean distance matrix in meters."""
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def _compute_bearing_matrix(self) -> np.ndarray:
        """Computes pairwise compass bearing angle in degrees from node i to node j.

        0° = North (+y), 90° = East (+x), 180° = South (-y), 270° = West (-x).
        """
        dx = self.coords[None, :, 0] - self.coords[:, None, 0]  # x_j - x_i
        dy = self.coords[None, :, 1] - self.coords[:, None, 1]  # y_j - y_i
        # atan2(dx, dy) yields compass bearing where 0 is North and 90 is East
        bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
        return bearing.astype(np.float32)

    def _build_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constructs edges based on the selected topology method."""
        directed_edges: set[Tuple[int, int]] = set()

        if self.method == "delaunay":
            tri = Delaunay(self.coords)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            u, v = simplex[i], simplex[j]
                            directed_edges.add((u, v))

        elif self.method == "knn":
            for i in range(self.num_nodes):
                # Sort neighbors by distance (excluding self)
                distances = self.dist_matrix[i].copy()
                distances[i] = np.inf
                nearest = np.argsort(distances)[: self.knn_k]
                for j in nearest:
                    directed_edges.add((i, j))
                    directed_edges.add((j, i))

        elif self.method == "threshold":
            if self.dist_threshold is None:
                raise ValueError("dist_threshold must be provided when method='threshold'")
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j and self.dist_matrix[i, j] <= self.dist_threshold:
                        directed_edges.add((i, j))

        elif self.method == "hybrid":
            # Delaunay edges combined with k-NN
            tri = Delaunay(self.coords)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            directed_edges.add((simplex[i], simplex[j]))

            for i in range(self.num_nodes):
                distances = self.dist_matrix[i].copy()
                distances[i] = np.inf
                nearest = np.argsort(distances)[: self.knn_k]
                for j in nearest:
                    directed_edges.add((i, j))
                    directed_edges.add((j, i))
        else:
            raise ValueError(f"Unknown graph topology method: {self.method}")

        # Filter by distance threshold if specified
        if self.dist_threshold is not None:
            directed_edges = {
                (u, v) for u, v in directed_edges if self.dist_matrix[u, v] <= self.dist_threshold
            }

        sorted_edges = sorted(list(directed_edges))
        if len(sorted_edges) == 0:
            raise ValueError("No edges generated under the specified graph criteria.")

        src = [e[0] for e in sorted_edges]
        dst = [e[1] for e in sorted_edges]
        edge_index = np.array([src, dst], dtype=np.int64)

        edge_dist = np.array([self.dist_matrix[u, v] for u, v in sorted_edges], dtype=np.float32)
        edge_bearing = np.array([self.bearing_matrix[u, v] for u, v in sorted_edges], dtype=np.float32)

        return edge_index, edge_dist, edge_bearing

    def get_pyg_data(self) -> Data:
        """Returns a PyTorch Geometric Data object with static graph topology."""
        return Data(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            edge_attr=torch.stack([self.edge_dist, self.edge_bearing], dim=-1),
            pos=self.pos,
            num_nodes=self.num_nodes,
        )

    def get_dense_adjacency(self) -> torch.Tensor:
        """Returns a dense [num_nodes, num_nodes] adjacency matrix with Gaussian weights."""
        adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        adj[self.edge_index[0], self.edge_index[1]] = self.edge_weight
        return adj

    def __repr__(self) -> str:
        return (
            f"WindTurbineGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.edge_index.shape[1]}, "
            f"method='{self.method}', "
            f"mean_dist={self.edge_dist.mean():.1f}m, "
            f"sigma={self.sigma:.1f}m)"
        )


def build_static_graph(
    locations_path: str = "data/raw/sdwpf_baidukddcup2022_turb_location.csv",
    method: Literal["delaunay", "knn", "threshold", "hybrid"] = "delaunay",
    knn_k: int = 6,
    dist_threshold: Optional[float] = None,
    sigma: Optional[float] = None,
) -> WindTurbineGraph:
    """Convenience factory function to instantiate a WindTurbineGraph."""
    return WindTurbineGraph(
        locations=locations_path,
        method=method,
        knn_k=knn_k,
        dist_threshold=dist_threshold,
        sigma=sigma,
    )
