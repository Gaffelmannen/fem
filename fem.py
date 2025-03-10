#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.linalg import solve
import yaml

class FiniteElementModel:

    def __init__(self, nodes, elements, material_properties, boundary_conditions):
        self.nodes = np.array(nodes) 
        self.elements = np.array(elements)  # List of (node1, node2, node3) tuples (triangular elements)
        self.material_properties = material_properties  # Dictionary of material properties (e.g., Young's modulus)
        self.boundary_conditions = boundary_conditions  # Dict with node indices and prescribed displacements
        self.displacements = None
        self.stresses = None
        self.global_stiffness_matrix = None

    def assemble_stiffness_matrix(self):
        num_nodes = len(self.nodes)
        self.global_stiffness_matrix = np.zeros((2*num_nodes, 2*num_nodes))
        # Loop over elements to assemble global stiffness matrix
        for element in self.elements:
            ke = self.element_stiffness_matrix(element)
            indices = np.array([(2*node, 2*node+1) for node in element]).flatten()
            for i in range(6):
                for j in range(6):
                    self.global_stiffness_matrix[indices[i], indices[j]] += ke[i, j]
    
    def element_stiffness_matrix(self, element):
        # Extract node coordinates for the element
        node_indices = element
        coords = self.nodes[node_indices]

        # Calculate area of the triangular element
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        A = 0.5 * np.linalg.det(np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ]))

        # Calculate B-matrix (strain-displacement matrix)
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B = (1 / (2 * A)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        # Material properties
        E = self.material_properties['E']
        nu = self.material_properties['nu']

        # Plane stress condition D-matrix
        D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

        # Calculate element stiffness matrix
        ke = A * B.T @ D @ B

        return ke

    def apply_boundary_conditions(self, external_forces):
        num_nodes = len(self.nodes)
        self.force_vector = np.zeros(2*num_nodes)
        for node, force in external_forces.items():
            self.force_vector[2*node] = force[0]  # Fx
            self.force_vector[2*node+1] = force[1]  # Fy
        for node, displacement in self.boundary_conditions.items():
            index_x = 2*node
            index_y = 2*node+1
            self.global_stiffness_matrix[index_x, :] = 0
            self.global_stiffness_matrix[index_y, :] = 0
            self.global_stiffness_matrix[:, index_x] = 0
            self.global_stiffness_matrix[:, index_y] = 0
            self.global_stiffness_matrix[index_x, index_x] = 1
            self.global_stiffness_matrix[index_y, index_y] = 1
            self.force_vector[index_x] = displacement[0]
            self.force_vector[index_y] = displacement[1]

    def solve(self, external_forces):
        self.assemble_stiffness_matrix()
        self.apply_boundary_conditions(external_forces)
        self.displacements = solve(self.global_stiffness_matrix, self.force_vector)

    def visualize_stress_distribution(self, material_name=""):
        if self.displacements is None:
            return
        
        element_stresses = []
        for element in self.elements:
            node_indices = np.array(element)
            coords = self.nodes[node_indices]
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            
            A = 0.5 * np.linalg.det(np.array([
                [1, x1, y1],
                [1, x2, y2],
                [1, x3, y3]
            ]))
            
            b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
            c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
            
            B = (1 / (2 * A)) * np.array([
                [b1, 0, b2, 0, b3, 0],
                [0, c1, 0, c2, 0, c3],
                [c1, b1, c2, b2, c3, b3]
            ])
            
            D = (self.material_properties['E'] / (1 - self.material_properties['nu']**2)) * np.array([
                [1, self.material_properties['nu'], 0],
                [self.material_properties['nu'], 1, 0],
                [0, 0, (1 - self.material_properties['nu']) / 2]
            ])
            
            element_displacements = self.displacements[np.repeat(node_indices, 2) * 2 + np.tile([0, 1], 3)]
            stress = D @ B @ element_displacements
            von_mises_stress = np.linalg.norm(stress)
            element_stresses.append(von_mises_stress)
        
        plt.figure(figsize=(5, 5))
        triangulation = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        plt.tripcolor(triangulation, element_stresses, shading='gouraud', cmap='jet')
        plt.colorbar(label='Von Mises Stress')
        plt.title(f"Stress Distribution - {material_name}")
        plt.savefig(f"img/fem_stress_{material_name}.png")

    def visualize(self, stress=False, material_name=""):
        plt.figure(figsize=(5, 5))
        triangulation = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        plt.triplot(triangulation, color="blue")
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], color="red")
        plt.title(f"Finite Element Mesh - {material_name}")
        plt.savefig(f"img/fem_mesh - {material_name}.png")

    def visualize_deformation(self, material_name=""):
        # Plot deformed shape if displacements are available
        if self.displacements is not None:
            scale = 1
            deformed_nodes = self.nodes + scale * self.displacements.reshape((-1, 2))
            deformed_triangulation = tri.Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1], self.elements)
            plt.triplot(deformed_triangulation, color="red")
            plt.title(f"Finite Element Mesh of {material_name} with Deformation")
            plt.savefig(f"img/fem_dispacement_{material_name}.png")


if __name__ == "__main__":

    # Read the material config file
    config = None

    # Material
    with open("materials.yml", "r") as f:
        config = yaml.safe_load(f)
    selected_material = config["materials"]["Wood"]

    # Material properties
    material_properties = {
        # Young's modulus in Pascals
        'E': float(selected_material["modulus"]),
        # Poisson's ratio
        'nu': float(selected_material["poisson_ratio"])
    }

    # Load the geometry file
    with open("geometry.yml", "r") as file:
        config = yaml.safe_load(file)
    
    # Nodes
    nodes = [tuple(node["coordinates"]) for node in config["geometry"]["nodes"]]

    # Define triangular elements by referencing nodes
    elements = [tuple(element["nodes"]) for element in config["geometry"]["elements"]]

    # Boundary conditions (Fixed on the left edge)
    boundary_conditions = {}
    for fixed in config["geometry"]["boundary_conditions"]["fixed_displacements"]:
        node = fixed["node"]
        ux = fixed.get("ux", None)
        uy = fixed.get("uy", None)
        boundary_conditions[node] = (ux, uy)

    # External forces (Horizontal force on the right edge)
    external_forces = {}
    for force in config["geometry"]["external_forces"]["forces"]:
        node = force["node"]
        fx = force.get("fx", 0)
        fy = force.get("fy", 0)
        external_forces[node] = (fx, fy)
    
    # Create and run the FEM model
    fem_model = FiniteElementModel(nodes, elements, material_properties, boundary_conditions)
    
    fem_model.solve(external_forces)

    fem_model.visualize(stress=False, material_name=selected_material["name"])
    fem_model.visualize_deformation(material_name=selected_material["name"])
    fem_model.visualize_stress_distribution(material_name=selected_material["name"])
