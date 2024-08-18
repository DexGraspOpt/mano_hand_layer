import numpy as np


def reconstruct_points(A, B, C, u, v, w):
    # Convert vertices to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # Reconstruct points
    P = u[:, None] * A + v[:, None] * B + w[:, None] * C
    return P


# Example usage
A = np.array([[0, 0, 0], [1, 0, 0]])
B = np.array([[1, 0, 0], [0, 1, 0]])
C = np.array([[0, 1, 0], [0, 0, 1]])
P = np.array([[0.3, 0.3, 0], [0.2, 0.2, 0.2]])



# Calculate barycentric coordinates
def barycentric_coordinates_3d_batch(A, B, C, P):
    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


u, v, w = barycentric_coordinates_3d_batch(A, B, C, P)

# Reconstruct the original points
reconstructed_P = reconstruct_points(A, B, C, u, v, w)
print("Reconstructed points:")
print(reconstructed_P)